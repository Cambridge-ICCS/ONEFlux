function [fOut, iOut] = identifyOutliers(xNormX, threshold)
    % identifyOutliers
    % Identifies outliers based on standardized scores.
        
    fOut = xNormX > threshold; 
    iOut = find(fOut); 
        
    end