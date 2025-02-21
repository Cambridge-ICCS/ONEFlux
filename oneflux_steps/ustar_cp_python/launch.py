import os
import glob
from pathlib import Path
from typing import Tuple, List, Union, Any
import numpy as np
import pandas as pd
import copy
from oneflux_steps.ustar_cp_python.cpdBootstrap import cpdBootstrapUStarTh4Season20100901

def launch(input_folder: str, output_folder: str) -> int:
    """
    Main function to perform U* threshold computation by Alan Barr.

    This function:
    - Calls checkPath to ensure the input and output folder paths are valid.
    - Searches for input files matching '*_qca_ustar_*.csv' in input_folder.
    - For each file, reads header info, parses data, checks if PPFD is valid or 
      needs to be derived from SW_IN, replaces missing values with NaN, ensures 
      key columns aren't empty, then computes time array and calls further 
      bootstrapping/assigning routines (cpdBootstrapUStarTh4Season20100901, 
      cpdAssignUStarTh20100901, etc.).
    - Saves results and updates an error code if any step fails. TODO: Implement saveResult.
    
    Parameters
    ----------
    :param input_folder: Path to the folder containing input files.
    :param output_folder: Path to the folder where output files will be saved.

    Returns
    -------
    :return: exitcode (0 if successful, 1 if any file failed to process).
    """
    exitcode = 0

    # Check and fix the folder paths if needed.
    input_folder, output_folder = checkPath(input_folder, output_folder)

    # Use 0-based indices instead of 1-based.
    USTAR_INDEX = 0
    NEE_INDEX = 1
    TA_INDEX = 2
    PPFD_INDEX = 3
    RG_INDEX = 4

    input_columns_names = ['USTAR', 'NEE', 'TA', 'PPFD_IN', 'SW_IN']

    print("\n\nUstar Threshold Computation by Alan Barr")
    print(f"input in {input_folder}")
    print(f"output in {output_folder}\n")

    # Find files matching the pattern "*_qca_ustar_*.csv"
    d = glob.glob(os.path.join(input_folder, "*_qca_ustar_*.csv"))
    error_str = []

    print(f"{len(d)} files founded.\n")

    
    for n in range(len(d)):

        print(f"processing n.{(n+1):02d}, {os.path.basename(d[n])}...", end="")

        # Try to open the file
        try:
            with open(d[n], 'r') as fid:
                # Read all lines
                dataset = fid.read().splitlines()
        except OSError:
            print("unable to open file")
            exitcode = 1
            continue

        # Parse the header
        errorCode, site, year, lat, lon, timezone, htower, timeres, sc_negl, notes = notValidHeader(dataset)
        if errorCode == 1:
            exitcode = 1
            continue

        
        i = 9
        while True:
            if i >= len(dataset):
                break
            # Check if the line starts with "notes" (case-insensitive, first 5 chars)
            if not dataset[i].lower().startswith('notes'):
                break
            temp = dataset[i].replace('notes,', '')
            # Prepending in a list context
            notes = [temp] + notes
            i += 1

        filename = os.path.basename(d[n])

        # Load the data
        header, data, columns_index = loadData(input_folder, filename, notes, input_columns_names)

        # Map column names to indices
        errorCode, columns_index = mapColumnNamesToIndices(header, input_columns_names, notes, columns_index)
        if errorCode == 1:
            exitcode = 1
            continue

        # Check if PPFD column exists (or is derived from RG)
        ppfd_from_rg, errorCode = ppfdColExists(PPFD_INDEX, columns_index, input_columns_names)
        if errorCode == 1:
            exitcode = 1
            continue

        # Extract columns using 0-based indexing
        uStar = data.iloc[:, columns_index[USTAR_INDEX]]
        NEE = data.iloc[:, columns_index[NEE_INDEX]]
        Ta = data.iloc[:, columns_index[TA_INDEX]]
        Rg = data.iloc[:, columns_index[RG_INDEX]]

        # Check if all PPFD values are invalid
        PPFD, ppfd_from_rg = areAllPpfdValuesInvalid(ppfd_from_rg, columns_index, PPFD_INDEX, data)

        # If PPFD should be derived from Rg
        if ppfd_from_rg == 1:
            PPFD = derivePpfdColFromRg(Rg)

        # Clear 'data'
        del data

        # Replace missing data (-9999) with NaN
        uStar, NEE, Ta, PPFD, Rg = setMissingDataNan(uStar, NEE, Ta, PPFD, Rg)

        # Check for empty or all-NaN columns
        errorCode = anyColumnsEmpty(uStar, NEE, Ta, Rg)
        if errorCode == 1:
            exitcode = 1
            continue

        # Create time array
        t = createTimeArray(uStar)

        # Flag nighttime periods
        fNight = Rg < 5

        # Copy of Ta
        T = Ta

        # Plot data (returns fPlot)
        # fPlot = plotData(t, uStar, NEE, Ta, PPFD, Rg)

        # Prepare for bootstrap program
        fPlot = 0
        cSiteYr = filename.replace(".txt", "")  # e.g. 'CACa1-2001'
        cSiteYr = cSiteYr.replace("_ut", "_barr")
        nBoot = 100

        # 4-season analysis
        Cp2, Stats2, Cp3, Stats3 = cpdBootstrapUStarTh4Season20100901(
            t, NEE, uStar, T, fNight, fPlot, cSiteYr, nBoot
        )

        Cp, n_val, tW, CpW, cMode, cFailure, fSelect, sSine, FracSig, FracModeD, FracSelect = \
            cpdAssignUStarTh20100901(Stats2, fPlot, cSiteYr)

        # Save result
        # 'clock' in MATLAB typically returns the current date/time. We'll pass Python's datetime now.
        from datetime import datetime
        current_time = datetime.now()
        # TODO: Implement saveResult
        error_str, cSiteYr, errorCode = saveResult(
            cFailure, cSiteYr, output_folder, site, year, Cp, current_time, notes
        )
        if errorCode == 1:
            exitcode = 1
            continue

        # Clear references to variables at the end of iteration
        del (uStar, cFailure, cMode, cSiteYr, fNight, fPlot, fSelect, n_val,
             nBoot, sSine, t, tW, Cp3, CpW, FracModeD, FracSelect, FracSig,
             NEE, PPFD, Rg, Stats2, Stats3, T, Ta, Cp, Cp2)

        print("done.")

    # End of for loop
    return exitcode


def checkPath(input_folder: str, output_folder: str) -> Tuple[str, str]:
    """
    checkPath

    Ensures that input_folder and output_folder are valid directory paths
    by adjusting them if necessary. Uses pathlib for cleaner path handling
    and a helper function _fixPath to avoid duplicating logic.

    Parameters
    ----------
    input_folder : str
        Path to the input folder, potentially relative or empty.
    output_folder : str
        Path to the output folder, potentially relative or empty.

    Returns
    -------
    input_folder_fixed : str
        Adjusted, absolute path for the input folder, guaranteed to
        end with the OS's default separator.
    output_folder_fixed : str
        Adjusted, absolute path for the output folder, guaranteed to
        end with the OS's default separator. Created if it does not
        already exist.
    """

    def _fixPath(folder: str) -> Path:
        """
        _fixPath

        Helper function that:
          1) If folder is empty, defaults to the current working directory.
          2) If folder is extremely short (< 2 chars), interprets it as a
             relative path from the current working directory.
          3) Returns a resolved (absolute) pathlib.Path object.

        Parameters
        ----------
        folder : str
            Potentially empty or relative folder path.

        Returns
        -------
        path_obj : Path
            A resolved (absolute) Path object.
        """
        if not folder:  # e.g. empty string
            path_obj = Path.cwd()
        elif len(folder) < 2:
            path_obj = Path.cwd() / folder
        else:
            path_obj = Path(folder)

        # Return the absolute, normalized path
        return path_obj.resolve()

    # Fix both paths using the helper
    input_path = _fixPath(input_folder)
    output_path = _fixPath(output_folder)

    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert the Path objects back to strings
    # Append the OS-specific separator at the end for consistency
    input_folder_fixed = str(input_path) + os.sep
    output_folder_fixed = str(output_path) + os.sep

    return input_folder_fixed, output_folder_fixed


def notValidHeader(dataset: List[str]) -> Tuple[int, str, str, str, str, str, str, str, str, str]:
    """
    notValidHeader

    Checks if the given dataset list of strings contains valid header lines
    for: site, year, lat, lon, timezone, htower, timeres, sc_negl, notes.
    If a required keyword is missing or mismatched, errorCode is set to 1
    and the function returns immediately.

    Parameters
    ----------
    dataset : List[str]
        A list of at least 9 strings, each expected to start with a keyword
        (e.g. "site,MySite", "year,2020", etc.).

    Returns
    -------
    errorCode : int
        0 if all keywords match, 1 if a keyword is missing or invalid.
    site : str
        Extracted site field (after removing "site," prefix), or "NaN" if invalid.
    year : str
        Extracted year field (after removing "year," prefix), or "NaN" if invalid.
    lat : str
        Extracted lat field, or "NaN" if invalid.
    lon : str
        Extracted lon field, or "NaN" if invalid.
    timezone : str
        Extracted timezone field, or "NaN" if invalid.
    htower : str
        Extracted tower height, or "NaN" if invalid.
    timeres : str
        Extracted time resolution, or "NaN" if invalid.
    sc_negl : str
        Extracted "sc_negl" field, or "NaN" if invalid.
    notes : str
        Extracted notes field, or "NaN" if invalid.

    Example
    -------
    >>> data = [
    ...     "site,MySite", "year,2021", "lat,52.0", "lon,113.0",
    ...     "timezone,UTC+7", "htower,20m", "timeres,30min",
    ...     "Sc_negl,True", "notes,Some notes here"
    ... ]
    >>> notValidHeader(data)
    (0, 'MySite', '2021', '52.0', '113.0', 'UTC+7', '20m', '30min', 'True', 'Some notes here')
    """


    # Initialize errorCode
    errorCode = 0

    # Default values for each field if extraction fails
    extractedValues = ["NaN"] * 9

    # Check dataset length
    if len(dataset) < 9:
        errorCode = 1
        return (errorCode, *extractedValues)

    # List of (required prefix, error message) for each line
    fields = ["site", "year", "lat", "lon", "timezone", "htower", "timeres", "sc_negl", "notes"]

    # Use a list comprehension to build the (prefix, error message) pairs
    fieldInfo = [(field, f"{field} keyword not found.") for field in fields]

    # A mutable container for an error flag
    errorState = [0]

    def _extractFieldAtIndex(
        dataList: List[str],
        index: int,
        prefix: str,
        errorMsg: str,
        errorFlag: List[int]
    ) -> str:
        """
        Extracts and returns the substring after prefix in dataList[index].
        If the line doesn't start with prefix (case-insensitive),
        sets errorFlag[0] to 1 and returns "NaN".
        """
        line = dataList[index]
        if not line.lower().startswith(prefix.lower()):
            print(errorMsg)
            errorFlag[0] = 1
            return "NaN"

        # Attempt splitting on the first comma
        parts = line.split(",", 1)
        if len(parts) < 2:
            print(errorMsg)
            errorFlag[0] = 1
            return "NaN"

        return parts[1]


    # Extract each field in a loop
    for i, (prefix, errMsg) in enumerate(fieldInfo):
        extractedValues[i] = _extractFieldAtIndex(dataset, i, prefix, errMsg, errorState)
        if errorState[0] == 1:
            errorCode = 1
            return (errorCode, *extractedValues)

    # If we get here, everything is valid
    return (errorCode, *extractedValues)


def loadData(
    input_folder: str,
    filename: str,
    notes: List[str],
    input_columns_names: List[str],
    *args
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    loadData

    Reads a CSV-like file and returns
    a 'header', 'data', and a 'columns_index' array initialized to -1.

    Parameters
    ----------
    input_folder : str
        Path to the input folder (e.g., '/path/to/').
    filename : str
        Name of the file to load (e.g., 'data.csv').
    notes : List[str]
        A list of strings considered part of the header offset (i.e.,
        their length is used to skip extra lines).
    input_columns_names : List[str]
        Names of columns we expect to map later (mirroring MATLAB usage).
    *args
        Variable-length argument list, currently discards uneeded argument passed in.

    Returns
    -------
    header : List[str]
        Lines read as the 'textdata' portion.
    data : np.ndarray
        2D numpy array of floats read from the file after skipping lines.
    columns_index : np.ndarray
        An array of length len(input_columns_names), filled with -1.

    Notes
    -----
    - Manually reads the file lines and splits out text vs. data.

    """

    # Calculate how many lines to skip
    header_rows = 9 + len(notes)
    data_path = os.path.join(input_folder, filename)

    # Read the data
    data = pd.read_csv(data_path, skiprows=header_rows, header=None)

    # Get the number of columns
    columns = data.columns
    num_columns = len(columns.tolist())

    # Read the header
    header = pd.read_csv(data_path, names=range(num_columns), nrows=header_rows, header=None)
 
    # Initialize columns_index to -1
    columns_index = np.full(len(input_columns_names), -1)

    return header, data, columns_index


def mapColumnNamesToIndices(
    header: pd.DataFrame,
    input_column_names: List[str],
    notes: List[str],  # Currently unused, but kept for future extension
    columns_index: List[int],
    *args: Any
) -> Tuple[int, List[int]]:
    """
    Maps column names from a header DataFrame to their respective indices.

    This function searches the last row of the `header` DataFrame for matches 
    against the list of `input_column_names`. If a match is found (case-insensitive), 
    the corresponding element in `columns_index` is set to the index of that column in `header`.
    
    A special case is handled by prepending 'itp' to `input_column_names` 
    for case-insensitive matching as well (e.g., 'NEE' matches 'NEE' or 'itpNEE').
    
    Parameters
    ----------
    header : pd.DataFrame
        A DataFrame containing column names in its last row.
    input_column_names : List[str]
        A list of column names to find in the `header`.
    notes : str
        Currently unused, but included for potential future needs.
    columns_index : List[int]
        A list of indices corresponding to each `input_column_name`. 
        Initially, each element is typically set to -1, indicating the column 
        has not been found yet.
    *args : Any
        Additional arguments (currently unused).
    
    Returns
    -------
    Tuple[int, List[int]]
        A tuple where the first element is an error code (0 if successful, 
        1 if a duplicate is detected) and the second element is the updated 
        `columns_index` list, mapping each `input_column_name` to its index 
        in the header.

    Notes
    -----
    - If any column in `input_column_names` is found multiple times in the header, 
      the function prints a message about duplication, sets `errorCode` to 1, 
      and returns immediately (with `columns_index` in its last valid state).
    - The function performs a case-insensitive comparison. If an input column 
      name is, for example, 'NEE', it will match 'NEE' or 'nee'. It will 
      also match 'itpNEE' if 'itp' is prefixed in the header.
    """

    errorCode = 0

    # Get the list of column names from the last row of the header
    column_names = header.iloc[-1, :].tolist()

    # Loop through each column name found in the header
    for i, column_name in enumerate(column_names):
        # Compare with the list of input_column_names
        for j, input_col_name in enumerate(input_column_names):
            # Case-insensitive match. Also check 'itp' prefix.
            if (input_col_name.lower() == column_name.lower() or 
                f'itp{input_col_name.lower()}' == column_name.lower()):
                # If columns_index[j] is already set, it's a duplicate
                if columns_index[j] != -1:
                    print(f"The column '{input_col_name}' is duplicated.")
                    errorCode = 1
                    return errorCode, columns_index
                else:
                    # Update columns_index to reflect this column's index in the header
                    columns_index[j] = i

    return errorCode, columns_index


def ppfdColExists(
    PPFD_INDEX: int,
    columns_index: List[int],
    input_columns_names: List[str]
) -> Tuple[int, int]:
    """
    Checks whether the PPFD column exists based on the given column indices.
    
    If a column index is -1 and it corresponds to the PPFD_INDEX, this function 
    sets ppfd_from_rg to 1. If a column index is -1 and does not correspond 
    to PPFD_INDEX, it reports that the column is not found and sets an error code.
    
    Parameters
    ----------
    :param PPFD_INDEX: Index of the PPFD column.
    :param columns_index: List of column indices.
    :param input_columns_names: List of column names.

    Returns
    -------
    :return: A tuple containing (ppfd_from_rg, errorCode).

    """
    # Make shallow copies
    columns_index_copy = columns_index[:]

    errorCode = 0
    on_error = 0
    ppfd_from_rg = 0

    for i in range(len(columns_index_copy)):

        if columns_index_copy[i] == -1:
            if i == PPFD_INDEX:
                ppfd_from_rg = 1
            else:
                print(f"column {input_columns_names[i]} not found!")
                on_error = 1

    if on_error == 1:
        errorCode = 1

    return ppfd_from_rg, errorCode


def areAllPpfdValuesInvalid(
    ppfd_from_rg: int,
    columns_index: List[int],
    PPFD_INDEX: int,
    data: pd.DataFrame,
) -> Tuple[Union[str, pd.Series], int]:
    """
    Checks if all PPFD values in the specified column are invalid (less than -9990).
    
    If ppfd_from_rg is 0, the function attempts to retrieve the PPFD column from the data 
    using the provided columns_index[PPFD_INDEX]. If all values in that column are invalid 
    (i.e., below -9990), ppfd_from_rg is set to 1.
    
    Parameters
    ----------
    :param ppfd_from_rg: Indicator whether PPFD is retrieved from RG (0 means not yet).
    :param columns_index: List of column indices (0-based).
    :param PPFD_INDEX: The index (within columns_index) that points to the PPFD column.
    :param data: A pandas DataFrame containing the data.

    Returns
    -------
    :return: A tuple containing:
             - PPFD: A string (if unchanged) or a pandas Series of PPFD data.
             - ppfd_from_rg: Possibly updated indicator (1 if all PPFD values are invalid).
    """

    # Make shallow copies of inputs
    columns_index = columns_index.flatten()
    PPFD = ''

    if ppfd_from_rg == 0:
        # Retrieve the PPFD column using 0-based indexing
        PPFD = data.iloc[:, columns_index[PPFD_INDEX]]
        PPFD = PPFD.to_numpy()
        # Identify invalid values
        q = PPFD[PPFD < -9990]
        # If all values are invalid, mark ppfd_from_rg
        if len(q) == len(PPFD):
            ppfd_from_rg = 1

    return PPFD, ppfd_from_rg


def derivePpfdColFromRg(Rg: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Derives the PPFD column from Rg using the formula PPFD = Rg * 2.24. 
    It prints a message indicating that PPFD is derived from SW_IN.
    Any values corresponding to Rg < -9990 are set to -9999 in PPFD.
    
    Parameters
    ----------
    :param Rg: A pandas Series or NumPy array containing Rg data.
    Returns
    -------
    :return: A pandas Series or NumPy array containing the derived PPFD values.
    """

    print("(PPFD_IN from SW_IN)...")
    PPFD = Rg * 2.24  # Create the PPFD array/Series

    # Identify indices where Rg is invalid
    p = Rg < -9990

    # Set invalid PPFD values
    PPFD[p] = -9999

    return PPFD


def setMissingDataNan(
    uStar: np.ndarray,
    NEE: np.ndarray,
    Ta: np.ndarray,
    PPFD: np.ndarray,
    Rg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replaces -9999 with NaN in each provided NumPy array.

    This function creates a copy of each input array so that the original
    data remains unchanged. It then replaces every occurrence of -9999 with
    NaN in each copied array, and returns the modified arrays in the same order
    as the inputs.

    Parameters
    ----------
    uStar : np.ndarray
        NumPy array of uStar values.
    NEE : np.ndarray
        NumPy array of NEE values.
    Ta : np.ndarray
        NumPy array of Ta values.
    PPFD : np.ndarray
        NumPy array of PPFD values.
    Rg : np.ndarray
        NumPy array of Rg values.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing five NumPy arrays with -9999 replaced by NaN.
    """
    # List of input arrays for iteration
    arrays = [uStar, NEE, Ta, PPFD, Rg]

    # Initialize a list to hold the modified copies of the arrays
    modified_arrays = []

    # Loop over each array, copy it, and replace -9999 with np.nan
    for arr in arrays:
        # Create a copy of the array to avoid modifying the original
        arr_copy = arr.copy()
        # Replace all occurrences of -9999 with np.nan
        arr_copy[arr_copy == -9999] = np.nan
        modified_arrays.append(arr_copy)

    # Return the modified arrays as a tuple in the same order as the inputs
    return tuple(modified_arrays)


def anyColumnsEmpty(
    uStar: Union[np.ndarray, pd.Series],
    NEE: Union[np.ndarray, pd.Series],
    Ta: Union[np.ndarray, pd.Series],
    Rg: Union[np.ndarray, pd.Series]
) -> int:
    """
    Checks for empty or entirely NaN columns and returns an error code if any are found.

    - If all values in NEE are NaN, prints a message and sets errorCode = 1.
    - If all values in uStar are NaN, prints a message and sets errorCode = 1.
    - If Ta is empty (length == 0), prints a message and sets errorCode = 1.
    - If Rg is empty (length == 0), prints a message and sets errorCode = 1.
    
    :param uStar: Array/Series of uStar values.
    :param NEE: Array/Series of NEE values.
    :param Ta: Array/Series of Ta values.
    :param Rg: Array/Series of Rg values.
    :return: errorCode (0 if all checks pass, 1 if any column is empty/invalid).
    """
    errorCode = 0

    # Check if all NEE values are NaN
    if np.isnan(NEE).sum() == len(NEE):
        print("NEE is empty!")
        errorCode = 1
        return errorCode

    # Check if all uStar values are NaN
    if np.isnan(uStar).sum() == len(uStar):
        print("uStar is empty!")
        errorCode = 1
        return errorCode

    # Check if Ta is empty
    if len(Ta) == 0:
        print("Ta is empty!")
        errorCode = 1
        return errorCode

    # Check if Rg is empty
    if len(Rg) == 0:
        print("Rg is empty!")
        errorCode = 1
        return errorCode

    return errorCode


def createTimeArray(uStar: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Creates an array 't' whose length is the same as the length of 'uStar'.
    It first determines 'nrPerDay' using len(uStar) % 365. If that result is 0,
    it instead uses len(uStar) % 364. The first value of t is set to 
    1 + (1 / nrPerDay), and each subsequent value increments by (1 / nrPerDay).

    Parameters
    ----------
    :param uStar: Array or Series whose length determines the size of 't'.
    Returns
    -------
    :return: A NumPy array 't' of the same length as 'uStar'.
    """

    # No modification to uStar, so no copy is required here.
    n = len(uStar)
    nrPerDay = n % 365
    if nrPerDay == 0:
        nrPerDay = n % 364

    # Create 't' as a 1D NumPy array of zeros
    t = np.zeros(n, dtype=float)

    # Avoid division by zero if 'nrPerDay' is 0
    if nrPerDay != 0:
        # Set the initial value
        t[0] = 1 + (1 / nrPerDay)

        # Populate the rest of t
        for n2 in range(1, n):
            t[n2] = t[n2 - 1] + (1 / nrPerDay)

    return t





def saveResult(*args):
    return None, None, None