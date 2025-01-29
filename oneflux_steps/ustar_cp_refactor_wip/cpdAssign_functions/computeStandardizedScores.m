function xNormX = computeStandardizedScores(x)
    % computeStandardizedScores
    % Standardizes the matrix x by its median and interquartile range.
        
    mx = nanmedian(x); 
    sx = fcNaniqr(x); 
    xNorm = (x - mx) ./ sx; 
    xNormX = max(abs(xNorm), [], 2); 
        
    end