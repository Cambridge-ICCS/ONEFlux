function [iSelect, nSelect, fSelect] = updateSelectedIndices(iSelect, iOut, fSelect, fOut)
    % updateSelectedIndices
    % Updates selected indices after removing outliers.
        
    iSelect = setdiff(iSelect, iOut); 
    nSelect = length(iSelect); 
    fSelect = fSelect & ~fOut;
        
    end