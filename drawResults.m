% Evalulate the output results
%% Initialization
clear all;
close all;

%% Load images & correct labels from the test set
imageDir = 'E:/Code/ObjectDetection/new_data/test/image';
imageNum = length(dir(fullfile(imageDir, '*.png')));
labelDir = 'E:/Code/ObjectDetection/new_data/test/label';
labelNum = length(dir(fullfile(labelDir, '*.txt')));
outputDir = 'E:/Code/ObjectDetection/new_data/test/output';
outputNum = length(dir(fullfile(outputDir, '*.txt')));

assert(labelNum == imageNum);
% assert(labelNum == outputNum);
fprintf('Found %d images(labels).\n', labelNum);

imageNum = 1;%For debug

%% Draw

% Overlap threshold for cars and pedestrians. 0.7 and 0.5
% Ref: http://www.cvlibs.net/datasets/kitti/eval_object.php
carTh = 0.7;
pedTh = 0.5;

% Minium bounding box height that we consider. 
% Ref: Readme file from development kit
minObjHeight = 25;

fprintf('Start drawing.\n');

% for i=1:imageNum
for i=262:280
    if mod(i,100) == 0
        fprintf('%d\n', i);
    end
    
    im = imread(sprintf('%s/%06d.png', imageDir, i-1));
    % Read labels
    corrObjects = readLabels(labelDir, i-1);
    corrObjNum = numel(corrObjects);
    outObjects = readOutputLabels(outputDir, i-1);
    outObjNum = numel(outObjects);
    % Increase if we find this object
    findObjects = zeros(corrObjNum, 1);
    
    outRect = [];
    foundRect = [];
    corrRect = [];
    
    % For each output objects
    for iOut = 1:outObjNum
        if strcmp(outObjects(iOut).type, 'Car') %&& abs(outObjects(iOut).score) > 1
            found = false;
            % Check each correct cars and vans
            for iCorr = 1:corrObjNum
                if strcmp(corrObjects(iCorr).type, 'Car') || strcmp(corrObjects(iCorr).type, 'Van')
                    ratio = rectOverlap(outObjects(iOut).y1, outObjects(iOut).x1, outObjects(iOut).y2, outObjects(iOut).x2, ...
                        corrObjects(iCorr).y1, corrObjects(iCorr).x1, corrObjects(iCorr).y2, corrObjects(iCorr).x2);
                    if ratio > carTh || (outObjects(iOut).y1 > corrObjects(iCorr).y1 && outObjects(iOut).x1 > corrObjects(iCorr).x1 && outObjects(iOut).y2 < corrObjects(iCorr).y2 && outObjects(iOut).x2 < corrObjects(iCorr).x2)
                        % Found a corresponding object.
                        findObjects(iCorr) = findObjects(iCorr) + 1;
                        found = true;
                    end
                end
            end
            % Draw a car:
            if found
                foundRect = [foundRect; outObjects(iOut).x1, outObjects(iOut).y1, ...
                outObjects(iOut).x2-outObjects(iOut).x1, outObjects(iOut).y2-outObjects(iOut).y1];
            else
                outRect = [outRect; outObjects(iOut).x1, outObjects(iOut).y1, ...
                    outObjects(iOut).x2-outObjects(iOut).x1, outObjects(iOut).y2-outObjects(iOut).y1];
            end
        elseif strcmp(outObjects(iOut).type, 'Pedestrian') %&& abs(outObjects(iOut).score) > 1
            found = false;
            % Check each correct pedestrians and sitting persons
            for iCorr = 1:corrObjNum
                if strcmp(corrObjects(iCorr).type, 'Pedestrian') || strcmp(corrObjects(iCorr).type, 'Person_sitting')
                    ratio = rectOverlap(outObjects(iOut).y1, outObjects(iOut).x1, outObjects(iOut).y2, outObjects(iOut).x2, ...
                        corrObjects(iCorr).y1, corrObjects(iCorr).x1, corrObjects(iCorr).y2, corrObjects(iCorr).x2);
                    if ratio > pedTh || (outObjects(iOut).y1 > corrObjects(iCorr).y1 && outObjects(iOut).x1 > corrObjects(iCorr).x1 && outObjects(iOut).y2 < corrObjects(iCorr).y2 && outObjects(iOut).x2 < corrObjects(iCorr).x2)
                        % Found a corresponding object.
                        findObjects(iCorr) = findObjects(iCorr) + 1;
                        found = true;
                    end
                end
            end
            % Draw a pedestrian:
            if found
                foundRect = [foundRect; outObjects(iOut).x1, outObjects(iOut).y1, ...
                outObjects(iOut).x2-outObjects(iOut).x1, outObjects(iOut).y2-outObjects(iOut).y1];
            else
                outRect = [outRect; outObjects(iOut).x1, outObjects(iOut).y1, ...
                    outObjects(iOut).x2-outObjects(iOut).x1, outObjects(iOut).y2-outObjects(iOut).y1];
            end
        end
    end
    
    % Accumulate TP and FN for this image
    for iCorr = 1:corrObjNum
        if corrObjects(iCorr).y2 - corrObjects(iCorr).y1 >= minObjHeight
            if strcmp(corrObjects(iCorr).type, 'Car') || (strcmp(corrObjects(iCorr).type, 'Van') && findObjects(iCorr) > 0)
                % Draw a correct car
                corrRect = [corrRect; corrObjects(iCorr).x1, corrObjects(iCorr).y1, ...
                    corrObjects(iCorr).x2-corrObjects(iCorr).x1, corrObjects(iCorr).y2-corrObjects(iCorr).y1];
            elseif strcmp(corrObjects(iCorr).type, 'Pedestrian') || (strcmp(corrObjects(iCorr).type, 'Person_sitting') && findObjects(iCorr) > 0)
                % Draw a correct pedestrian
                corrRect = [corrRect; corrObjects(iCorr).x1, corrObjects(iCorr).y1, ...
                    corrObjects(iCorr).x2-corrObjects(iCorr).x1, corrObjects(iCorr).y2-corrObjects(iCorr).y1];
            end
        end
    end
    
    shapeInserter1 = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom', 'CustomBorderColor', uint8([0 255 0]), 'LineWidth', 6);
    shapeInserter2 = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom', 'CustomBorderColor', uint8([255 0 0]), 'LineWidth', 2);
    shapeInserter3 = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom', 'CustomBorderColor', uint8([0 0 255]), 'LineWidth', 4);
    
    if size(corrRect, 1) > 0
        J = step(shapeInserter1, im, int32(corrRect));
        if size(outRect, 1) > 0
            K = step(shapeInserter2, J, int32(outRect));
            if size(foundRect, 1) > 0
                L = step(shapeInserter3, K, int32(foundRect));
                currFig = imshow(L);
            else
                currFig = imshow(K);
            end
        elseif size(foundRect, 1) > 0
            L = step(shapeInserter3, K, int32(foundRect));
            currFig = imshow(L);
        else
            currFig = imshow(J);
        end
        title(num2str(i-1));
        waitfor(currFig);
    end
end
