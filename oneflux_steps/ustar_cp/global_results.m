function global_results()
    global launchstruct;
    launchstruct.input_folder = '';
    launchstruct.output_folder = '';
    launchstruct.d_array = [];
    launchstruct.dataset = struct();
    launchstruct.notes = '';
    launchstruct.imported_data = struct();
    launchstruct.header = [];
    launchstruct.data = [];
    launchstruct.columns_index = [];
    launchstruct.uStar = []; % see line 232
    launchstruct.NEE = []; % see line 232
    launchstruct.Ta = []; % see line 232
    launchstruct.Rg = []; % see line 232
    launchstruct.PPFD = [];
    launchstruct.t = [];
    launchstruct.T = [];
    launchstruct.fNight = [];
    launchstruct.fPlot = [];
    launchstruct.cSiteYr = [];
    launchstruct.nBoot = [];
    launchstruct.Cp2 = [];
    launchstruct.Stats2 = [];
    launchstruct.Cp3 = [];
    launchstruct.Stats3 = [];
    launchstruct.Cp = [];
    launchstruct.n = [];
    launchstruct.tW = [];
    launchstruct.CpW = [];
    launchstruct.cMode = [];
    launchstruct.cFailure = [];
    launchstruct.fSelect = [];
    launchstruct.sSine = [];
    launchstruct.FracSig = [];
    launchstruct.FracModeD = [];
    launchstruct.FracSelect = [];
    launchstruct.exitcode = [];

    global cpdAssignStruct;
    cpdAssignStruct.CpA = [];
    cpdAssignStruct.nA = [];
    cpdAssignStruct.CpW = [];
    cpdAssignStruct.fSelect = [];
    cpdAssignStruct.cMode = "";
    cpdAssignStruct.cFailure = "";
    cpdAssignStruct.sSine = [];
    cpdAssignStruct.FracSig = [];
    cpdAssignStruct.FracModeD = [];
    cpdAssignStruct.fModeE = [];
    cpdAssignStruct.FracSelect = [];
    cpdAssignStruct.nDim = [];
    cpdAssignStruct.nSelectN = [];
    cpdAssignStruct.c2 = [];
    cpdAssignStruct.cic2 = [];
    cpdAssignStruct.iTry = [];
    cpdAssignStruct.nTry = [];
    cpdAssignStruct.iCp = [];
    cpdAssignStruct.nCp = [];
    cpdAssignStruct.iNS = [];
    cpdAssignStruct.nNs = [];
    cpdAssignStruct.iSig = [];
    cpdAssignStruct.nSig = [];
    cpdAssignStruct.iModeE = [];
    cpdAssignStruct.nModeE = [];
    cpdAssignStruct.iModeD = [];
    cpdAssignStruct.nModeD = [];
    cpdAssignStruct.iSelect = [];
    cpdAssignStruct.nSelect = [];
    cpdAssignStruct.cMode = [];
    cpdAssignStruct.x = [];
    cpdAssignStruct.FracSig_2 = [];
	cpdAssignStruct.FracModeD_2 = [];
	cpdAssignStruct.FracSelect_2  = [];
    cpdAssignStruct.cFailure_2 = [];
    cpdAssignStruct.mtSelect = [];
    cpdAssignStruct.CpSelect = [];
    cpdAssignStruct.xBins = [];
    cpdAssignStruct.n = [];
    cpdAssignStruct.tW = [];
    cpdAssignStruct.CpW = [];

    global BS4S;
    % Variables from cpdBootstrapUStarTh4Season20100901
    BS4S.nPerSeason = [];
    BS4S.ntN = [];
    BS4S.ntNee = [];
    BS4S.Stats2 = [];
    BS4S.Stats3 = [];
    BS4S.xCp3 = [];
    BS4s.xStats3 = [];

end
