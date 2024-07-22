# Functional Flow Diagram for `cpdFindChangePoint20100901` Function

## Start
|
|-- **Initialize Function**
|   - Define function with inputs: `xx`, `yy`, `fPlot`, `cPlot`
|
|-- **Initialize Outputs**
|   - Set `Cp2`, `Cp3` to NaN
|   - Initialize `s2` and `s3` structures with default NaN values
|
|-- **Exclude Missing Data**
|   - Reshape `xx` and `yy` to column vectors
|   - Find and exclude NaN values in `xx` and `yy`
|   - Calculate number of valid data points `n`
|   - If `n` is less than 10, return
|
|-- **Exclude Extreme Linear Regression Outliers**
|   - Perform linear regression on `x` and `y` to get regression coefficients `a`
|   - Calculate predicted values `yHat` and residuals `dy`
|   - Calculate mean `mdy` and standard deviation `sdy` of residuals
|   - Find and exclude outliers beyond `ns` (4) standard deviations
|   - Calculate number of valid data points `n`
|   - If `n` is less than 10, return
|
|-- **Compute Null Hypothesis Models**
|   - Compute mean of `y` as `yHat2` and `SSERed2`
|   - Perform linear regression on `x` and `y` to get `yHat3` and `SSERed3`
|   - Set `nRed2 = 1`, `nFull2 = 2`, `nRed3 = 2`, `nFull3 = 3`
|
|-- **Compute F Scores**
|   - Initialize `MT`, `Fc2`, `Fc3` arrays with NaN values
|   - Set `nEndPtsN = 3` and calculate `nEndPts`
|   - Loop through each data point to compute F scores:
|   |   - Fit 2-parameter model and compute `Fc2`
|   |   - Fit 3-parameter model and compute `Fc3`
|
|-- **Assign Change Points**
|   - Find `Fmax2` and `iCp2`, set `xCp2`
|   - Perform linear regression for 2-parameter model and calculate `yHat2`
|   - Calculate p-value `p2` and assign `Cp2` if significant
|   - Find `Fmax3` and `iCp3`, set `xCp3`
|   - Perform linear regression for 3-parameter model and calculate `yHat3`
|   - Calculate p-value `p3` and assign `Cp3` if significant
|
|-- **Assign Values to s2 and s3**
|   - Check if `iCp2` is within valid range, if so, assign values to `s2`
|   - Check if `iCp3` is within valid range, if so, assign values to `s3`
|
|-- **Plot Results (if enabled)**
|   - If `fPlot` is 1:
|   |   - Plot `x`, `y`, `yHat2`, `yHat3`, `xCp2`, `xCp3`
|   |   - Set plot title and adjust plot limits
|   |   - Format plot appearance
|
|-- **End**
