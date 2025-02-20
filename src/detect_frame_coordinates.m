% filepath: detect_frame_coordinates.m
function frameCoordinates = detect_frame_coordinates(imagePath)
% Detects the frame coordinates in an image automatically.
%
%   Args:
%       imagePath (char): Path to the image file.
%
%   Returns:
%       frameCoordinates (numeric): Coordinates of the frame [x1, y1, x2, y2],
%                                  or NaN if frame detection fails.

try
    % Load the image
    img = imread(imagePath);

    % Convert to grayscale
    gray = rgb2gray(img);

    % Edge detection (Canny)
    edges = edge(gray, 'Canny');

    % Hough transform
    [H, theta, rho] = hough(edges);
    peaks = houghpeaks(H, 10, 'threshold', ceil(0.3*max(H(:))));
    lines = houghlines(edges, theta, rho, peaks, 'FillGap', 50, 'MinLength', 100);

    if isempty(lines)
        fprintf('No lines detected.\n');
        frameCoordinates = NaN;
        return;
    end

    % Filter lines (horizontal and vertical)
    horizontalLines = [];
    verticalLines = [];
    for k = 1:length(lines)
        angle = abs(lines(k).theta);
        if angle < 45 || angle > 135 % Roughly vertical
            verticalLines = [verticalLines, lines(k)];
        else % Roughly horizontal
            horizontalLines = [horizontalLines, lines(k)];
        end
    end

    if isempty(horizontalLines) || isempty(verticalLines)
        fprintf('Not enough horizontal or vertical lines detected.\n');
        frameCoordinates = NaN;
        return;
    end

    % Find extreme lines
    topLine = find_extreme_line(horizontalLines, 'min', 'y');
    bottomLine = find_extreme_line(horizontalLines, 'max', 'y');
    leftLine = find_extreme_line(verticalLines, 'min', 'x');
    rightLine = find_extreme_line(verticalLines, 'max', 'x');

    % Calculate intersection points
    topLeft = line_intersection(topLine, leftLine);
    topRight = line_intersection(topLine, rightLine);
    bottomLeft = line_intersection(bottomLine, leftLine);
    bottomRight = line_intersection(bottomLine, rightLine);

    if any(isnan([topLeft, topRight, bottomLeft, bottomRight]))
        fprintf('Could not find all intersection points.\n');
        frameCoordinates = NaN;
        return;
    end

    % Extract coordinates
    x1 = round(topLeft(1));
    y1 = round(topLeft(2));
    x2 = round(bottomRight(1));
    y2 = round(bottomRight(2));

    frameCoordinates = [x1, y1, x2, y2];

catch ME
    fprintf('An error occurred: %s\n', ME.message);
    frameCoordinates = NaN;
end
end

function extremeLine = find_extreme_line(lines, minOrMax, xOrY)
% Finds the line with the minimum or maximum x or y coordinate.
if isempty(lines)
    extremeLine = NaN;
    return;
end

if strcmp(xOrY, 'x')
    coords = arrayfun(@(line) (line.point1(1) + line.point2(1)) / 2, lines);
else
    coords = arrayfun(@(line) (line.point1(2) + line.point2(2)) / 2, lines);
end

if strcmp(minOrMax, 'min')
    [~, idx] = min(coords);
else
    [~, idx] = max(coords);
end

extremeLine = lines(idx);
end

function intersection = line_intersection(line1, line2)
% Calculates the intersection point of two lines.
if isnan(line1) || isnan(line2)
    intersection = [NaN, NaN];
    return;
end

x1 = line1.point1(1);
y1 = line1.point1(2);
x2 = line1.point2(1);
y2 = line1.point2(2);

x3 = line2.point1(1);
y3 = line2.point1(2);
x4 = line2.point2(1);
y4 = line2.point2(2);

denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
if denom == 0
    intersection = [NaN, NaN]; % Lines are parallel
    return;
end

px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

intersection = [px, py];
end