import numpy as np
from typing import Dict, List
from typing import Tuple
from oneflux_steps.ustar_cp_python.utilities import dot, intersect, round_up
from oneflux_steps.ustar_cp_python.cpd_evaluate_functions import cpdEvaluateUStarTh4Season20100901

def cpdBootstrapUStarTh4Season20100901(t: np.ndarray, NEE: np.ndarray, uStar: np.ndarray, T: np.ndarray, fNight: np.ndarray, fPlot: int, cSiteYr: str, nBoot: int) -> Tuple[np.ndarray, List[List[List[Dict[str, float]]]], np.ndarray, List[List[List[Dict[str, float]]]]]:
    """
    Bootstrap the uStarTh for a season.

    Args:
        t (np.ndarray): Time vector.
        NEE (np.ndarray): Net Ecosystem Exchange values.
        uStar (np.ndarray): u* values.
        T (np.ndarray): Temperature values.
        fNight (np.ndarray): Day/night flags.
        fPlot (int): Plot flag.
        cSiteYr (str): Site year.
        nBoot (int): Number of bootstraps.
        *args (Any): Additional arguments.

    Returns:
        Tuple: Cp2, Stats2, Cp3, Stats3

    Notes:
     	cpdBootstrapUStarTh4Season20100901 is a simplified operational version of 

     	20100901 changes 3A 4I to 2A 3I as sggested by Xiaolan Wang. 	
	
     	20100408 replaces 20100318, updating ChPt function from:
     	FindChangePointMod3LundReeves2002_20100318 to
     	FindChangePointMod2A4ILundReeves2002_20100408.
     	which: adds back A model, makes a correction to the significance test,
     	and adds a comparison of the 4- versus 3-parameter models. 
     	and adds a comparison of the 4- versus 3-parameter models. 

     	20100318 new version with small tweaks to FindChangePoint
     	also adds mT and ciT to Stats structures.
     
     	is a new working implementation of Alan's u*Th evaluation algorithm
     	based on Lund and Reeves' (2002) modified by Wang's (2003) change-point
     	detection algorithm. 
     
     	Relationship to other programs:
     
     	Called by batchNewNacpEstimateUStarTh_Moving_Mod3LundChPt_20100115 
     		- script which identifies which sites to process 
     	Calls newNacpEvaluateUStarTh_MovingStrat_20100114
     		- function that processes an individual year of data, using 
     		  FindChangePointMod3LundReeves2002_20091204
     	
     	This implementation may supplant all previous versions. 
     
     	It uses moving windows of fixed size to evaluate seasonal variation.  
     
     	Three combinations of stratification and pooling are implemented.  
     	 - All begin with 2D (time x temperature) stratification 
     	   (moving-window time x n temperature classes within each window). 
     	 - Two (W and A) add normalization and pooling.  
     
     	1. Method S (full stratification) 
     		estimates the change-points for each of the strata 
     		(nWindows x nTClasses) with no need for normalization.  
     	2. Method W (pooling within time windows) 
     		begins with a variant of S but pools the temperature classes 
     		within each time window before estimating one change-point per window. 
     		Before pooling, the binned mean NEE data within each temperature class 
     		are normalized against the 80th NEE percentile for that class. 
     	3. Method A (pooling to annual) 
     		further pools the pooled normalized data from W over all time windows 
     		before estimating a single change-point per year. 
      
     	The detailed analysis parameters are output in a Stats structured
     	record. 

     	Written originally in MATLAB by Alan Barr 15 Jan 2010. 

    """
    # Define constants
    nSeasons = 4
    nStrataX = 8

    # Initialize parameters
    nt = len(t)
    iNight = get_iNight(fNight)
    updated_uStar = update_uStar(uStar)
    ntN = get_ntN(t, nSeasons)

    # Generate itNee and ntNee
    itNee = get_itNee(NEE, uStar, T, iNight)
    ntNee = len(itNee)

    Cp2 = setup_Cp(nSeasons, nStrataX, nBoot)
    Cp3 = setup_Cp(nSeasons, nStrataX, nBoot)
    Stats2 = setup_Stats(nBoot, nSeasons, nStrataX)
    Stats3 = setup_Stats(nBoot, nSeasons, nStrataX)

    if ntNee >= ntN:
        # Boot-strapping
        for iBoot in range(nBoot):
            it = generate_rand_int_array(nt)
            # Old comment: ntNee=sum(ismember(it,itNee));
            if iBoot > 0:
                fPlot = 0

            xCp2, xStats2, xCp3, xStats3 = cpdEvaluateUStarTh4Season20100901(
                t[it], NEE[it], updated_uStar[it], T[it], fNight[it], fPlot, cSiteYr
            )

            Cp2[:, :, iBoot] = xCp2
            Stats2[:, :, iBoot] = xStats2
            Cp3[:, :, iBoot] = xCp3
            Stats3[:, :, iBoot] = xStats3

    return Cp2, Stats2, Cp3, Stats3

def generate_statsMT() -> Dict[str, float]:
    """
    Initialize a stats structure with NaN values for predefined fields.

    Returns:
        Dict[str, float]: Initialized stats dictionary with NaN values.
    """
    fields = [
        "n", "Cp", "Fmax", "p", "b0", "b1", "b2", "c2",
        "cib0", "cib1", "cic2", "mt", "ti", "tf",
        "ruStarVsT", "puStarVsT", "mT", "ciT"
    ]

    # Initialize all fields with NaN
    stats_mt = {field: np.nan for field in fields}

    return stats_mt

def generate_rand_int_array(n: int) -> np.ndarray:
    """
    Generate a random integer array of size n,
    sorted in ascending order, with numbrers
    ranging between 0 and (n-1)

    Args:
        n (int): Size of the array.

    Returns:
        np.ndarray: Random integer array.
    """
    return np.sort(np.random.randint(0, n, n))
    

def setup_Stats(n_boot: int, n_seasons: int, n_strata_x: int) -> List[List[List[Dict[str, float]]]]|dict[str, float]:
    """
    Initialize the Stats structure based on input dimensions.

    This function is correct where the args are all integers > 1.

    Args:
        n_boot (int): Number of bootstraps.
        n_seasons (int): Number of seasons.
        n_strata_x (int): Number of strata in X direction.

    Returns:
        List[List[List[Dict[str, float]]]]: Preallocated stats structure.
    """
    args_list = [n_boot, n_seasons, n_strata_x]
    for n in args_list:
        if n < 2:
            raise ValueError(f'Function "setup_stats" has been passed an argument \
                             with value {n}. This may lead undesired behaviour')

    # Preallocate stats array
    stats = [[[generate_statsMT() for _ in range(n_boot)]
              for _ in range(n_strata_x)]
             for _ in range(n_seasons)]

    return stats

def setup_Cp(nSeasons : int, nStrataX : int, nBoot : int) -> np.ndarray:
    """
    Initialize the Cp structure based on input dimensions.
    
    Args:
        nSeasons (int): Number of seasons.
        nStrataX (int): Number of strata in X direction.
        nBoot (int): Number of bootstraps.
    
    Returns:
        np.ndarray: Preallocated Cp structure.
    """
    return dot(np.nan, np.ones([nSeasons, nStrataX, nBoot]))

def get_itNee(NEE : np.ndarray, uStar : np.ndarray, T : np.ndarray, iNight : np.ndarray) -> np.ndarray:
    """
    Get the indices of the NEE values (that intersect with the night time values); excluding
    any indices with a NaN value in NEE, uStar, or T.

    Args:
        NEE (np.ndarray): Net Ecosystem Exchange values.
        uStar (np.ndarray): uStar values.
        T (np.ndarray): Temperature values
        iNight (np.ndarray): Night time values.

    Returns:
        np.ndarray: Indices of the NEE values.
    """
    itNee = np.where(np.logical_not(np.isnan(NEE + uStar + T)))[0]
    # Interect the arrays of itNee and itNight
    itNee = intersect(itNee, iNight)
    return itNee

def get_ntN(t : np.ndarray, nSeasons : int) -> int:
    """
    Get the number of points in the season.

    Args:
        t (np.ndarray): Time vector.
        nSeasons (int): Number of seasons.

    Returns:
        int: Number of points in the season.
        
    """
    nStrataN = 4 # Local variable, used to calculate ntN
    nBins = 50   # Local variable, used to calculate ntN

    nPerBin = get_nPerBin(t)

    # Calculate ntN based on nStrataN, nBins, and nPerBin
    nPerSeason = nStrataN * nBins * nPerBin
    ntN = nSeasons * nPerSeason
    return ntN

def update_uStar(uStar : np.ndarray) -> np.ndarray:
    """
    Update uStar values in the input array, replacing
    all values below 0 and above 4 with NaN.
    """
    updated_ustar = uStar.copy() # Initialize to same size as input
    iOut = np.where(np.logical_or(uStar < 0, uStar > 4))[0]
    updated_ustar[iOut] = np.nan
    return updated_ustar

def get_iNight(fNight : np.ndarray) -> np.ndarray:
    """
    Get the indices of the night time values.

    Args:
        fNight (np.ndarray): Night time values.         
    Returns:  
        np.ndarray: Indices of the night time values.

    """
    return np.where(fNight)[0]

def get_nPerBin(t : np.ndarray) -> int:
    """
    Get the number of points per bin.

    Args: 
        t (np.ndarray): Time vector.
    Returns:  
        int: Number of points per bin (within a day)
    """
    nPerDay = get_nPerDay(t)
    if 24 == nPerDay:
        return 3
    elif 48 == nPerDay:
        return 5
    else:
        # Default case
        return 5

def get_nPerDay(t):
    """
    Get the number of points per day.
    
    Args:
        t (np.ndarray): Time vector.

    Returns:  
        int: Number of points per day.
    """
    return round_up(1 / np.nanmedian(np.diff(t)))