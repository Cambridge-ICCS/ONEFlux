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
        