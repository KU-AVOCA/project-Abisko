% %% register_thermal_folder.m
% % Batch-register thermal images using a precomputed transformation.
% % - Loads all .mat files from an input folder (each containing 'thermal_image')
% % - Applies geometric correction using a saved transformation (tform_thermalRGB.mat)
% % - Saves the registered image as both .mat (variable: 'thermal_image_registered') and .npy
% % - No cropping is performed

% % --- User parameters ---
% input_folder = 'I:\SCIENCE-IGN-ALL\AVOCA_Group\2_Shared_folders\5_Projects\2025Abisko\Tower thermal images\preview\all\matimages';      % Folder with .mat files (each with 'thermal_image')
% output_mat_folder = 'I:\SCIENCE-IGN-ALL\AVOCA_Group\2_Shared_folders\5_Projects\2025Abisko\Tower thermal images\preview\all\registered';   % Output folder for .mat files
% % output_npy_folder = 'PATH/TO/YOUR/OUTPUT/npy';   % Output folder for .npy files
% tform_file = 'tform_thermalRGB.mat';             % Path to transformation file

% % Create output folders if they don't exist
% if ~exist(output_mat_folder, 'dir'), mkdir(output_mat_folder); end
% % if ~exist(output_npy_folder, 'dir'), mkdir(output_npy_folder); end

% % Load transformation and RGB image size
% S = load(tform_file); % Should contain 'tform' and 'rgbImageSize'
% tform = S.tform;
% rgbImageSize = S.rgbImageSize;

% % Get list of .mat files in input folder
% mat_files = dir(fullfile(input_folder, 'West-facing*.mat'));

% fprintf('Found %d .mat files in %s\n', numel(mat_files), input_folder);

% for k = 1:numel(mat_files)
%     fname = mat_files(k).name;
%     fpath = fullfile(input_folder, fname);
%     data = load(fpath);
%     if ~isfield(data, 'thermal_image')
%         fprintf('Skipping %s (no variable "thermal_image")\n', fname);
%         continue
%     else 
%         fprintf('Processing %s..., %d out of %d\n', fname, k, numel(mat_files));
%     end
%     thermal_image = data.thermal_image;

%     % Register image
%     thermal_image_registered = imwarp(thermal_image, tform, 'OutputView', imref2d(rgbImageSize));

%     % Save as .mat
%     out_mat = fullfile(output_mat_folder, fname);
%     save(out_mat, 'thermal_image_registered');

%     % Save as .npy (requires npy-matlab: https://github.com/kwikteam/npy-matlab)
%     % out_npy = fullfile(output_npy_folder, [erase(fname, '.mat') '.npy']);
%     % try
%     %     writeNPY(thermal_image_registered, out_npy);
%     % catch
%     %     warning('Could not save %s as .npy (is npy-matlab installed and on path?)', fname);
%     % end

%     fprintf('Processed %s\n', fname);
% end

% fprintf('Batch registration complete.\n');

%% thermalRGBregistration_batch.m
% Batch-register thermal images using a precomputed transformation (with parallel processing).
% - Loads all .mat files from an input folder (each containing 'thermal_image')
% - Applies geometric correction using a saved transformation (tform_thermalRGB.mat)
% - Saves the registered image as .mat (variable: 'thermal_image_registered')
% - Uses parallel processing for faster execution
% - No cropping is performed

% --- User parameters ---
input_folder = 'I:\SCIENCE-IGN-ALL\AVOCA_Group\2_Shared_folders\5_Projects\2025Abisko\Tower thermal images\preview\all\matimages';      % Folder with .mat files (each with 'thermal_image')
output_mat_folder = 'I:\SCIENCE-IGN-ALL\AVOCA_Group\2_Shared_folders\5_Projects\2025Abisko\Tower thermal images\preview\all\registered';   % Output folder for .mat files
tform_file = 'tform_thermalRGB.mat';             % Path to transformation file

% Create output folders if they don't exist
if ~exist(output_mat_folder, 'dir'), mkdir(output_mat_folder); end

% Load transformation and RGB image size
S = load(tform_file); % Should contain 'tform' and 'rgbImageSize'
tform = S.tform;
rgbImageSize = S.rgbImageSize;

% Get list of .mat files in input folder
mat_files = dir(fullfile(input_folder, 'West-facing*.mat'));
num_files = numel(mat_files);

fprintf('Found %d .mat files in %s\n', num_files, input_folder);

% Start parallel pool if not already started
pool = gcp('nocreate'); % Get current pool without creating new one
if isempty(pool)
    pool = parpool(); % Create default parallel pool
    fprintf('Started parallel pool with %d workers\n', pool.NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', pool.NumWorkers);
end

% Process files in parallel
tic; % Start timer
parfor k = 1:num_files
    fname = mat_files(k).name;
    fpath = fullfile(input_folder, fname);
    
    % Load thermal image
    data = load(fpath);
    if ~isfield(data, 'thermal_image')
        fprintf('Skipping %s (no variable "thermal_image")\n', fname);
        continue
    end
    thermal_image = data.thermal_image;

    % Register image
    thermal_image_registered = imwarp(thermal_image, tform, ...
        'OutputView', imref2d(rgbImageSize), "FillValues", NaN);

    % convert temperature from celsius to kelvin
    index = isnan(thermal_image_registered);
    thermal_image_registered = thermal_image_registered + 273.15;
    % scale to uint16
    thermal_image_registered = uint16(thermal_image_registered * 100); % e.g., 3000 = 30.00 °C
    thermal_image_registered(index) = 0; % set NaNs to 0
    % Save as .mat
    out_mat = fullfile(output_mat_folder, fname);
    parsave(out_mat, thermal_image_registered); % Custom save function for parfor
    
    % Progress indicator
    if mod(k, 10) == 0
        fprintf('Processed %d out of %d files\n', k, num_files);
    end
end

elapsed_time = toc; % End timer
fprintf('Batch registration complete in %.2f seconds (%.2f files/sec)\n', ...
    elapsed_time, num_files/elapsed_time);

% Helper function for saving in parfor loop
function parsave(fname, thermal_image_registered)
    save(fname, 'thermal_image_registered');
end