function sSine = fitAnnualSineCurve(mt, Cp, iSelect)
  % TODO: remove logging
    %  fid = fopen('mlog.txt', 'a');
    % fprintf(fid, '-------------\n');
    % fprintf(fid, 'mt shape = %d\n', size(mt));
    % fprintf(fid, 'mt = %s\n', mat2str(mt, 4));

    % fprintf(fid, 'Cp shape = %d\n', size(Cp));
    % fprintf(fid, 'Cp = %s\n', mat2str(Cp, 4));

    % fprintf(fid, 'iSelect shape = %d\n', size(iSelect));
    % fprintf(fid, 'iSelect = %s\n', mat2str(iSelect, 4));

    % fitAnnualSineCurve
    % Fits an annual sine curve to the selected data points and returns
    % the sine coefficients and r-squared value.
        
    % Initial Guess for Sine Coefficients
    bSine = [1, 1, 1]; 
        
    % Fit Sine Curve Using Nonlinear Regression
    dis = statset();
    dis.Display = 'iter';
    % fprintf(fid, '#Starting nlinfit...\n');
    % printf size of inputs
    % fprintf(fid, 'mt(iSelect) shape = %d\n', size(mt(iSelect)));
    % fprintf(fid, 'Cp(iSelect) shape = %d\n', size(Cp(iSelect)));

    bSine = nlinfit(mt(iSelect), Cp(iSelect), 'fcEqnAnnualSine', bSine); 
    % fprintf(fid, 'popt shape = %d\n', size(bSine));
    % fprintf(fid, 'popt = %s\n', mat2str(bSine, 4));
    
        
    % Calculate Predicted Values and R-Squared
    predictedCp = fcEqnAnnualSine(bSine, mt(iSelect)); 
    r2 = fcr2Calc(Cp(iSelect), predictedCp); 
        
    % Adjust Phase and Wrap to One Year
    bSine(3) = mod(bSine(3), 365.25);
        
    % Return Fitted Sine Coefficients and R-Squared
    sSine = [bSine, r2]; 
    % fprintf(fid, 'bSine = %s\n', mat2str(bSine));
    % fprintf(fid, 'r2 = %s\n', mat2str(r2));

    % fprintf(fid, 'sSine = %s\n', mat2str(sSine));
    % fclose(fid);
        
    end