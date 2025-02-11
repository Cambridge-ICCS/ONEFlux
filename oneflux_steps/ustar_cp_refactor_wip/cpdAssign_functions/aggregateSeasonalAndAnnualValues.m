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