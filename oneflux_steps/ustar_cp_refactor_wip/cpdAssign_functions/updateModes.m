function [iModeX, nModeX] = updateModes(fModeX, iOut)
    % updateModes
    % Recalculates mode indices after removing outliers.
        
    fModeX(iOut) = NaN; 
    iModeX = find(fModeX == 1); 
    nModeX = length(iModeX); 
        
    end
    