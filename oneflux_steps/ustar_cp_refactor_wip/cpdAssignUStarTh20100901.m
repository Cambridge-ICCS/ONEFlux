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

fid = fopen('mlog.txt', 'a');
fprintf(fid, '* Cp = %d\n', size(Cp));
fprintf(fid, '* b1 = %d\n', size(b1));
fprintf(fid, '* c2 = %d\n', size(c2));
fclose(fid);
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
metadata = struct();
metadata.siteFile = 'US-Cbo'; % Site file name
metadata.oneFluxDir = '/Users/dorchard/Documents/iccs/ONEFlux';
metadata.relArtifactsDir = 'tests/test_artifacts';
metadata.frequency = 10; % Log every 10th call
metadata.offset = 0; % Start logging from the first call

%standardizedScores = computeStandardizedScores(regressionMatrix);

metadata.inputNames = {'regressionMatrix'};
metadata.outputNames = {'standardizedScores'};
standardizedScores = logFuncResult('log.json', @computeStandardizedScores, metadata, regressionMatrix)

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
    
        
        
        
           

