%% thermalRGBcontrolpoints.m
% Script for registering a thermal image to an RGB image using manually selected control points.
% Steps:
%   1. Load images and preprocess
%   2. (Optional) Select control points using cpselect
%   3. Refine control points using cross-correlation
%   4. Fit geometric transformation (polynomial, projective, or PWL)
%   5. Register and crop images to minimize edge effects
%   6. Visualize and save results

%% 1. Load and preprocess images
% Load transformation and control points if available
if isfile("tform_thermalRGB.mat")
    load("tform_thermalRGB.mat");
end

% Load thermal image (.mat file should contain variable 'img')
thermalMat = load("I:\SCIENCE-IGN-ALL\AVOCA_Group\2_Shared_folders\5_Projects\2025Abisko\Tower thermal images\preview\all\matimages\West-facing_2023-03-20_10.30.02.mat");
thermalImage = thermalMat.img;

% Remove redundant sensor columns (136 to 142)
thermalImage(:, 136:142) = [];

% Normalize thermal image for display
thermalImageNorm = mat2gray(thermalImage);

% Load RGB image and convert to grayscale for control point selection
rgbImage = imread("I:\SCIENCE-IGN-ALL\AVOCA_Group\1_Personal_folders\1_Simon\1_Abisko\6_Tower_Data\Tower RGB images\1 Data\1 Years\West-facing\2023\IMG_6534.JPG");
rgbImageGray = im2gray(rgbImage);

%% 2. (Optional) Select control points manually
% Uncomment the following line to launch cpselect for manual point selection:
% h = cpselect(thermalImageNorm, rgbImageGray);

%% 3. Refine control points using cross-correlation
% movingPoints: points in thermal image (moving)
% fixedPoints: points in RGB image (fixed)
% These should be defined or loaded from previous session
movingPointsAdjusted = cpcorr(movingPoints, fixedPoints, thermalImage, rgbImageGray);

% Visualize control points on both images
figure;
imshow(rgbImage);
hold on;
plot(fixedPoints(:,1), fixedPoints(:,2), 'xr');
title('Fixed Points on RGB Image');

figure;
imshow(thermalImageNorm);
hold on;
plot(movingPointsAdjusted(:,1), movingPointsAdjusted(:,2), 'xr');
title('Adjusted Moving Points on Thermal Image');

%% 4. Fit geometric transformation
% Choose transformation type: 'polynomial', 'projective', or 'pwl'
% Polynomial degree 3 is used here (works well for moderate distortions)
tform = fitgeotrans(movingPointsAdjusted, fixedPoints, "polynomial", 3);

% Alternative options:
% tform = fitgeotrans(movingPointsAdjusted, fixedPoints, "projective");
% tform = fitgeotrans(movingPointsAdjusted, fixedPoints, "pwl");

%% 5. Register and crop images to minimize edge effects
% Register thermal image to RGB image
rgbImageSize = size(rgbImage);
thermalRegistered = imwarp(thermalImage, tform, OutputView=imref2d(rgbImageSize));

% Visualize registration results
figure;
imshowpair(rgbImage, thermalRegistered, "blend");
title('Registered Thermal and RGB Images (Blended)');

figure;
imshow(mat2gray(thermalRegistered));
title('Registered Thermal Image');

figure;
imshow(rgbImage);
title('Original RGB Image');

% Crop images to remove edge artifacts
% Remove top 800 rows, then add a 50-pixel buffer on all sides
cropTop = 800;
buffer = 50;

thermalRegisteredCropped = thermalRegistered(cropTop+1:end, :);
rgbImageCropped = rgbImage(cropTop+1:end, :, :);

thermalRegisteredCropped = thermalRegisteredCropped(buffer+1:end-buffer, buffer+1:end-buffer);
rgbImageCropped = rgbImageCropped(buffer+1:end-buffer, buffer+1:end-buffer, :);

figure;
imshowpair(rgbImageCropped, thermalRegisteredCropped, "falsecolor");
title('Cropped Registered Images (Falsecolor)');

%% 6. Save control points and transformation
save("tform_thermalRGB.mat", "tform", "movingPointsAdjusted", ...
    "fixedPoints", "movingPoints", "rgbImageSize");

