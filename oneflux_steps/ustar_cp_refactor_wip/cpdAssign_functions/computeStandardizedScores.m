function xNormX = computeStandardizedScores(x)
    % computeStandardizedScores
    % Standardizes the matrix x by its median and interquartile range.
    % 
    % x : N x M (can contain NaNs)
    %    - mx = nanmedian(x)   -> 1 x M median (column-wise)
    %    - sx = fcNaniqr(x)    -> 1 x M IQR (column-wise)
    %    - xNorm = (x - mx) ./ sx
    %    - xNormX = max(abs(xNorm), [], 2) -> N x 1 
    %
    % By default, if a row has a NaN in one of its columns, that
    % can lead to xNormX(row)=NaN, since max(abs(...)) sees NaN.
    
    % if x is all nans write to a file called log.txt the number of nans
    % and the number of rows in x
    if all(isnan(x))
         fid = fopen('mlog.txt', 'a');
         fprintf(fid, 'All NaNs in x: %d rows\n', size(x));
         fclose(fid);
    end

    
    mx = nanmedian(x); 
    sx = fcNaniqr(x); 
    xNorm = (x - mx) ./ sx; 
    % returns a column vector containing the maximum value of each row.
    xNormX = max(abs(xNorm), [], 2); 
end