function [CpA, nA, tW, CpW, cMode, cFailure, fSelect, sSine, FracSig, FracModeD, FracSelect] = cpdAssignUStarTh20100901(Stats, fPlot, cSiteYr, varargin) 

    % Initialize Variables
    CpA = []; nA = []; tW = []; CpW = []; fSelect = []; cMode = ''; cFailure = ''; sSine = []; 
    FracSig = []; FracModeD = []; FracSelect = []; 
    
    % Decode JSON if Needed
    for i = 1:length(varargin)
        a = varargin{i};
        if iscell(a) && strcmp(a{1}, 'jsondecode')
            for j = 2:length(a)
                if a{j} == 1
                    Stats = jsondecode(Stats);
                end
            end
        end
    end
    
    % Determine Window Sizes
    nDim = ndims(Stats); 
    if nDim == 2
        [nWindows, nBoot] = size(Stats);
        nStrata = 1; nStrataN = 0.5; 
    elseif nDim == 3
        [nWindows, nStrata, nBoot] = size(Stats); 
        nStrataN = 1; 
    else
        cFailure = 'Stats must be 2D or 3D.'; 
        return; 
    end
    
    % Set Reference Values
    nWindowsN = 4; 
    nSelectN = nWindowsN * nStrataN * nBoot; 
    
    % Preallocate Outputs
    CpA = NaN(nBoot, 1); 
    nA = NaN(nBoot, 1); 
    tW = NaN(nWindows, 1); 
    CpW = NaN(nWindows, 1);
    
    % Extract Variables from Stats Structure
    cVars = {'mt', 'Cp', 'b1', 'c2', 'cib1', 'cic2', 'p'}; 
    nVars = length(cVars); 
    
    for i = 1:nVars
        cv = cVars{i}; 
        eval([cv ' = fcReadFields(Stats, ''' cv ''');']); 
        
        if strcmp(cv, 'mt')
            xmt = mt; 
        elseif strcmp(cv, 'Cp')
            xCp = Cp; 
        end
        
        eval([cv ' = fcx2colvec(' cv ');']); 
    end
    
    % Identify Significant Change Points
    pSig = 0.05; 
    fP = p <= pSig; 
    
    % Identify Model Type
    if sum(~isnan(c2)) == 0
        nPar = 2; 
        c2 = zeros(size(b1)); 
        cic2 = zeros(size(b1)); 
    else
        nPar = 3; 
    end
    
    % Classify Significant Change Points
    iTry = find(~isnan(mt)); 
    nTry = length(iTry); 
    
    %iCp = find(~isnan(b1 + c2 + Cp)); 
    %nCp = length(iCp); 
    
    %iNS = find(fP == 0 & ~isnan(b1 + c2 + Cp)); 
    %nNS = length(iNS); 
    
    iSig = find(fP == 1 & ~isnan(b1 + c2 + Cp)); 
    nSig = length(iSig); 
    
    iModeE = find(fP == 1 & b1 < c2); 
    nModeE = length(iModeE); 
    
    iModeD = find(fP == 1 & b1 >= c2); 
    nModeD = length(iModeD); 
    
    if nModeD >= nModeE
        iSelect = iModeD; 
        cMode = 'D'; 
    else
        iSelect = iModeE; 
        cMode = 'E'; 
    end
    
    nSelect = length(iSelect); 
    
    % Update Selection Flags
    fSelect = false(size(fP)); 
    fSelect(iSelect) = true; 
    
    fModeD = NaN(size(fP)); 
    fModeD(iModeD) = 1; 
    
    fModeE = NaN(size(fP)); 
    fModeE(iModeE) = 1; 

    
    FracSig = nSig / nTry; 
    FracModeD = nModeD / nSig; 
    FracSelect = nSelect / nTry; 
    
    % Abort if Too Few Selections
    if FracSelect < 0.10
        cFailure = 'Less than 10% successful detections.'; 
        return; 
    end
    
    % Configure Regression Matrix
    if nPar == 2
        x = [Cp, b1, cib1]; 
        %nx = 3; 
    else
        x = [Cp, b1, c2, cib1, cic2]; 
        %nx = 5; 
    end

    % Exclude Outliers
    xNormX = computeStandardizedScores(x); 
    
    % Identify Outliers
    [fOut, iOut] = identifyOutliers(xNormX, 5);

    % Update Selected Indices
    [iSelect, nSelect, fSelect] = updateSelectedIndices(iSelect, iOut, fSelect, fOut);
    
    % Update Modes
    [iModeD, nModeD] = updateModes(fModeD,iOut);
    [iModeE, ~] = updateModes(fModeE,iOut);

    % Recalculate Significant Indices
    iSig = union(iModeD, iModeE); 
    nSig = length(iSig);

    FracSig = nSig / nTry; 
    FracModeD = nModeD / nSig; 
    FracSelect = nSelect / nTry;
    
    if nSelect < nSelectN
        cFailure = sprintf('Too few selected change points: %g/%g', nSelect, nSelectN); 
        return; 
    end
    
    % Aggregate Seasonal and Annual Values
    [CpA, nA, ~] = aggregateSeasonalAndAnnualValues(xCp, iSelect, nDim, nWindows, nStrata, nBoot); 

    
    % Aggregate Seasonal Means
    [tW, CpW] = aggregateSeasonalMeans(mt, Cp, xmt, iSelect, nWindows, nStrata, nBoot);
    
    % Fit Annual Sine Curve
    sSine = fitAnnualSineCurve(mt, Cp, iSelect);

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
    
        
        
        
           

