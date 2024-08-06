function result = test_wrapper(input_path, output_path)
    % Initialize the global struct to return intermediate variables
    global_results();

    % This function passes input and output paths to the launch.m function
    result = launch(input_path, output_path);

    global launchstruct
    global cpdAssignStruct
    global BS4S

    dataList = {'launchStruct',launchstruct; 'cpdAssignStruct',cpdAssignStruct; 'BS4s',BS4S};

    for n = 1:size(dataList, 1)

        structName = dataList{n,1};
        currentStruct = dataList{n, 2};
    
        % Encode the struct to a JSON string
        jsonStruct = jsonencode(currentStruct);

        %specify the directory path and file name
        directoryPath = output_path;
        fileName = strcat(structName, '.json');

        %Concatenate the directory path and file name
        filePath = fullfile(directoryPath, fileName);

        % Write the JSON string to a file
        fid = fopen(filePath, 'w');
        if fid == -1, error('Cannot create JSON file'); end
        fwrite(fid, jsonStruct, 'char');
        fclose(fid);
    end


end
