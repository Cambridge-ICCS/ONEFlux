function [annualChangePoint, numAnnualSelected, seasonalTimeWindow, seasonalChangePoint, ...
    dominantMode, failureMessage, selectedPointsFlag, sineCurve, ...
    fractionSignificant, fractionModeD, fractionSelected] = cpdAssignUStarTh20100901(Stats, plotFlag, siteYearText, varargin)

% Initialize Variables
annualChangePoint = []; numAnnualSelected = []; seasonalTimeWindow = []; seasonalChangePoint = [];
selectedPointsFlag = []; dominantMode = ''; failureMessage = ''; sineCurve = [];
fractionSignificant = []; fractionModeD = []; fractionSelected = [];

% Decode JSON if Needed
for i = 1:length(varargin)
    arg = varargin{i};
    if iscell(arg) && strcmp(arg{1}, 'jsondecode')
        for j = 2:length(arg)
            if arg{j} == 1
                Stats = jsondecode(Stats);
            end
        end
    end
end

% Determine Window Sizes
numDimensions = ndims(Stats); 
if numDimensions == 2
    [numWindows, numBootstraps] = size(Stats);
    numTemperatureStrata = 1; temperatureStrataFactor = 0.5; 
elseif numDimensions == 3
    [numWindows, numTemperatureStrata, numBootstraps] = size(Stats); 
    temperatureStrataFactor = 1; 
else
    failureMessage = 'Stats must be 2D or 3D.'; 
    return; 
end

% Set Reference Values
referenceWindows = 4; 
requiredSelectionCount = referenceWindows * temperatureStrataFactor * numBootstraps; 

% Preallocate Outputs
annualChangePoint = NaN(numBootstraps, 1); 
numAnnualSelected = NaN(numBootstraps, 1); 
seasonalTimeWindow = NaN(numWindows, 1); 
seasonalChangePoint = NaN(numWindows, 1);

% Extract Variables from Stats Structure
variableNames = {'mt', 'Cp', 'b1', 'c2', 'cib1', 'cic2', 'p'}; 
numVariables = length(variableNames); 

for i = 1:numVariables
    variableName = variableNames{i}; 
    eval([variableName ' = fcReadFields(Stats, ''' variableName ''');']); 
    
    if strcmp(variableName, 'mt')
        measurementTime = mt; 
    elseif strcmp(variableName, 'Cp')
        changePoint = Cp; 
    end
    
    eval([variableName ' = fcx2colvec(' variableName ');']); 
end

% Identify Significant Change Points
significanceThreshold = 0.05; 
significantFlag = p <= significanceThreshold; 

% Identify Model Type
if sum(~isnan(c2)) == 0
    numParameters = 2; 
    c2 = zeros(size(b1)); 
    cic2 = zeros(size(b1)); 
else
    numParameters = 3; 
end

% Classify Significant Change Points
validMeasurementIndices = find(~isnan(mt)); 
numValidMeasurements = length(validMeasurementIndices); 

nonSignificantIndices = find(significantFlag == 0 & ~isnan(b1 + c2 + Cp)); 
numNonSignificant = length(nonSignificantIndices); 

significantIndices = find(significantFlag == 1 & ~isnan(b1 + c2 + Cp)); 
numSignificant = length(significantIndices); 

modeEIndices = find(significantFlag == 1 & b1 < c2); 
numModeE = length(modeEIndices); 

modeDIndices = find(significantFlag == 1 & b1 >= c2); 
numModeD = length(modeDIndices); 

% Select Dominant Mode
if numModeD >= numModeE
    selectedIndices = modeDIndices; 
    dominantMode = 'D'; 
else
    selectedIndices = modeEIndices; 
    dominantMode = 'E'; 
end

numSelected = length(selectedIndices); 

% Update Selection Flags
selectedPointsFlag = false(size(significantFlag)); 
selectedPointsFlag(selectedIndices) = true; 

modeDFlag = NaN(size(significantFlag)); 
modeDFlag(modeDIndices) = 1; 

modeEFlag = NaN(size(significantFlag)); 
modeEFlag(modeEIndices) = 1; 

fractionSignificant = numSignificant / numValidMeasurements; 
fractionModeD = numModeD / numSignificant; 
fractionSelected = numSelected / numValidMeasurements; 

% Abort if Too Few Selections
if fractionSelected < 0.10
    failureMessage = 'Less than 10% successful detections.'; 
    return; 
end

% Configure Regression Matrix
if numParameters == 2
    regressionMatrix = [Cp, b1, cib1]; 
else
    regressionMatrix = [Cp, b1, c2, cib1, cic2]; 
end

% Exclude Outliers
standardizedScores = computeStandardizedScores(regressionMatrix); 
[outlierFlag, outlierIndices] = identifyOutliers(standardizedScores, 5);

[selectedIndices, numSelected, selectedPointsFlag] = ...
    updateSelectedIndices(selectedIndices, outlierIndices, selectedPointsFlag, outlierFlag);

[modeDIndices, numModeD] = updateModes(modeDFlag, outlierIndices);
[modeEIndices, ~] = updateModes(modeEFlag, outlierIndices);

significantIndices = union(modeDIndices, modeEIndices); 
numSignificant = length(significantIndices);

fractionSignificant = numSignificant / numValidMeasurements; 
fractionModeD = numModeD / numSignificant; 
fractionSelected = numSelected / numValidMeasurements;

if numSelected < requiredSelectionCount
    failureMessage = sprintf('Too few selected change points: %g/%g', numSelected, requiredSelectionCount); 
    return; 
end

% Aggregate Seasonal and Annual Values
[annualChangePoint, numAnnualSelected, ~] = ...
    aggregateSeasonalAndAnnualValues(changePoint, selectedIndices, numDimensions, numWindows, numTemperatureStrata, numBootstraps); 

% Aggregate Seasonal Means
[seasonalTimeWindow, seasonalChangePoint] = ...
    aggregateSeasonalMeans(mt, Cp, measurementTime, selectedIndices, numWindows, numTemperatureStrata, numBootstraps);

% Fit Annual Sine Curve
sineCurve = fitAnnualSineCurve(mt, Cp, selectedIndices);

end


function sSine = fitAnnualSineCurve(mt, Cp, iSelect)
    % fitAnnualSineCurve
    % Fits an annual sine curve to the selected data points and returns
    % the sine coefficients and r-squared value.
        
    % Initial Guess for Sine Coefficients
    bSine = [1, 1, 1]; 
        
    % Fit Sine Curve Using Nonlinear Regression
    bSine = nlinfit(mt(iSelect), Cp(iSelect), 'fcEqnAnnualSine', bSine); 
        
    % Calculate Predicted Values and R-Squared
    predictedCp = fcEqnAnnualSine(bSine, mt(iSelect)); 
    r2 = fcr2Calc(Cp(iSelect), predictedCp); 
        
    % Adjust Phase and Wrap to One Year
    bSine(3) = mod(bSine(3), 365.25); 
        
    % Return Fitted Sine Coefficients and R-Squared
    sSine = [bSine, r2]; 
        
    end

function [tW, CpW] = aggregateSeasonalMeans(mt, Cp, xmt, iSelect, nWindows, nStrata, nBoot)
    % aggregateSeasonalMeans
    % Aggregates seasonal means for time and change points.
        
    % Calculate Median Number of Windows
    nW = nanmedian(sum(~isnan(reshape(xmt, nWindows, nStrata * nBoot)))); 
        
    % Sort Selected Measurements
    [mtSelect, i] = sort(mt(iSelect)); 
    CpSelect = Cp(iSelect(i)); 
        
    % Define Bins Based on Percentiles
    xBins = prctile(mtSelect, 0:(100/nW):100); 
        
    % Aggregate Seasonal Means Using fcBin
    [~, tW, CpW] = fcBin(mtSelect, CpSelect, xBins, 0); 
        
    end
        
function [CpA, nA, xCpSelect] = aggregateSeasonalAndAnnualValues(xCp, iSelect, nDim, nWindows, nStrata, nBoot)
    % aggregateSeasonalAndAnnualValues
    % Aggregates seasonal and annual change point values based on selection.

    % Initialize Selection Array
    xCpSelect = NaN(size(xCp)); 
        
    % Assign Selected Change Points
    xCpSelect(iSelect) = xCp(iSelect); 
    xCpGF = xCpSelect; 
        
    % Aggregate Values Based on Dimensions
    if nDim == 2
        CpA = nanmean(xCpGF); 
        nA = sum(~isnan(xCpSelect)); 
    elseif nDim == 3
        CpA = nanmean(reshape(xCpGF, nWindows * nStrata, nBoot)); 
        nA = sum(~isnan(reshape(xCpSelect, nWindows * nStrata, nBoot))); 
    else
        error('Invalid number of dimensions: Expected 2D or 3D Stats.');
    end
    
    end

function xNormX = computeStandardizedScores(x)
    % computeStandardizedScores
    % Standardizes the matrix x by its median and interquartile range.
        
    mx = nanmedian(x); 
    sx = fcNaniqr(x); 
    xNorm = (x - mx) ./ sx; 
    xNormX = max(abs(xNorm), [], 2); 
        
    end

function [fOut, iOut] = identifyOutliers(xNormX, threshold)
    % identifyOutliers
    % Identifies outliers based on standardized scores.
        
    fOut = xNormX > threshold; 
    iOut = find(fOut); 
        
    end

function [iSelect, nSelect, fSelect] = updateSelectedIndices(iSelect, iOut, fSelect, fOut)
    % updateSelectedIndices
    % Updates selected indices after removing outliers.
        
    iSelect = setdiff(iSelect, iOut); 
    nSelect = length(iSelect); 
    fSelect = fSelect & ~fOut;
        
    end

function [iModeX, nModeX] = updateModes(fModeX, iOut)
    % updateModes
    % Recalculates mode indices after removing outliers.
        
    fModeX(iOut) = NaN; 
    iModeX = find(fModeX == 1); 
    nModeX = length(iModeX); 
        
    end
    
        
        
        
           

