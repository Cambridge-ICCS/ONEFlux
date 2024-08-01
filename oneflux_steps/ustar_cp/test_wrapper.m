function result = test_wrapper(input_path, output_path)
    % Initialize the global struct to return intermediate variables
    initialize_global_struct();

    % This function passes input and output paths to the launch.m function
    result = launch(input_path, output_path);
end
