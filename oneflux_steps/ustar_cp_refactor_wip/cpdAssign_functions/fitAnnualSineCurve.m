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