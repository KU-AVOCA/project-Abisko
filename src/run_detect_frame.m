% filepath: run_detect_frame.m
% Example usage:
imagePath = 'path/to/your/image.jpg'; % Replace with the actual path to your image
frameCoordinates = detect_frame_coordinates(imagePath);

if ~any(isnan(frameCoordinates))
    fprintf('Frame coordinates: [%d, %d, %d, %d]\n', frameCoordinates(1), frameCoordinates(2), frameCoordinates(3), frameCoordinates(4));
else
    fprintf('Frame detection failed.\n');
end
% Display the image with the detected frame
img = imread(imagePath);
if ~any(isnan(frameCoordinates))
    % Ensure frame coordinates are within image bounds
    frameCoordinates(1:2) = max(frameCoordinates(1:2), 1); % Ensure top-left corner is within bounds
    frameCoordinates(3) = min(frameCoordinates(3), size(img, 2)); % Ensure width does not exceed image width
    frameCoordinates(4) = min(frameCoordinates(4), size(img, 1)); % Ensure height does not exceed image height

    % Draw a rectangle around the detected frame
    imshow(img);
    hold on;
    rectangle('Position', [frameCoordinates(1), frameCoordinates(2), frameCoordinates(3) - frameCoordinates(1), frameCoordinates(4) - frameCoordinates(2)], 'EdgeColor', 'r', 'LineWidth', 2);
    hold off;
else
    imshow(img);
    title('Frame detection failed.');
end