import numpy as np
from typing import Dict, List
from oneflux_steps.ustar_cp_python.utilities import dot, intersect, squeeze, diff, nanmedian

def cpdBootstrapUStarTh4Season20100901(*args, **kwargs):
    """
    cpdBootstrapUStarTh4Season20100901: Bootstrap the uStarTh for a season

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        None
    """
    # TODO: implement this function
    return None

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


def setup_Stats(n_boot: int, n_seasons: int, n_strata_x: int, **kwargs) -> List[List[List[Dict[str, float]]]]|dict[str, float]:
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
    stats = [[[generate_statsMT() for _ in range(n_strata_x)]
              for _ in range(n_seasons)]
             for _ in range(n_boot)]

    return stats

def setup_Cp(nSeasons=None, nStrataX=None, nBoot=None):
    # TODO: check definition, may need to use the definition in utils.py
    return dot(np.nan, np.ones([nSeasons, nStrataX, nBoot]))

# TODO: rough attempt in np
def get_itNee(NEE : np.ndarray, uStar : np.ndarray, T : np.ndarray, iNight : np.ndarray) -> np.ndarray:
    itNee = np.where(np.logical_not(np.isnan(NEE + uStar + T)))
    # Interect the arrays of itNee and itNight
    itNee = intersect(itNee, iNight)
    return itNee

def get_ntN(t, nSeasons):
    """
    Get the number of points in the season.
    """
    nStrataN = 4 # Local variable, used to calculate ntN
    nBins = 50   # Local variable, used to calculate ntN

    nPerBin = get_nPerBin(t);

    # Calculate ntN based on nStrataN, nBins, and nPerBin
    nPerSeason = nStrataN * nBins * nPerBin;
    ntN = nSeasons * nPerSeason;
    return ntN

def update_uStar(uStar : np.ndarray) -> np.ndarray:
    """
    Update uStar values in the input array, replacing
    all values below 0 and above 4 with NaN.
    """
    # TODO: check whether we need to change the indexing
    updated_ustar = uStar.copy() # Initialize to same size as input
    iOut = np.where(uStar < 0 | uStar > 4);
    updated_ustar[iOut] = np.nan()
    return updated_ustar

def get_iNight(fNight):
    return np.arange(0, len(fNight))

def get_nPerBin(t):    
    nPerDay = get_nPerDay(t)
    if 24 == nPerDay:
        return 3
    elif 48 == nPerDay:
        return 5
    else:
        # Default case
        return 5

def get_nPerDay(t):
    return round(1 / nanmedian(np.diff(t)))