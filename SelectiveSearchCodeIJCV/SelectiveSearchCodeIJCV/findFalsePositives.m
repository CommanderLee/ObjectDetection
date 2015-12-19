% Find the False Positives. 
% Using Selective Search -> Test with trained SVM -> Find FP.
% Modified from demo.m - Zhen
% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%% Initialization
clear all;
close all;

addpath('Dependencies');

fprintf('Demo of how to run the code for:\n');
fprintf('   J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   IJCV 2013\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made.\n');
%     fprintf('   
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%% Load images from the train set
imageDir = 'E:/Code/ObjectDetection/new_data/train/image';
imageNum = length(dir(fullfile(imageDir, '*.png')));
labelDir = 'E:/Code/ObjectDetection/new_data/train/label';
labelNum = length(dir(fullfile(labelDir, '*.txt')));

assert(labelNum == imageNum);
fprintf('Found %d images(labels).\n', labelNum);

% imageNum = 1;%For debug

%% Set parameters
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.8;

fpDir = 'E:/Code/ObjectDetection/crop/false_positive';
% Overlap threshold for cars and pedestrians. 0.7 and 0.5
% Use 0.3 and 0.2 for generating.
% Ref: http://www.cvlibs.net/datasets/kitti/eval_object.php
carTh = 0.3;
pedTh = 0.2;
% Minium bounding box height that we consider. 
% Ref: Readme file from development kit
minObjHeight = 25;

% Change fpNum to determin how many FP images we want from each img.
fpNum = 2;
fpIndex = 0;
rng(1);

load('svm_car.mat');
load('svm_ped.mat');
load('svm_car_ped.mat');
cellSize = [8 8];
imageSize = [64 64];

%% Start searching
tic;
fprintf('Looking for false positives.\n');
% For each image, choose 'bgNum' background:
for i=1:imageNum
    if mod(i,100) == 0
        fprintf('%d\n', i);
    end
    
    im = imread(sprintf('%s/%06d.png', imageDir, i-1));

    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);

    % Read labels
    objects = readLabels(labelDir, i-1);
    
    % Calculate overlap and save FP
    boxNum = size(boxes, 1);
    randBoxIndex = randperm(boxNum);
    counter = 0;
    % For each bounding box, check overlap with positive case
    for b = 1:boxNum
        currBox = boxes(randBoxIndex(b), :);
        if currBox(3) - currBox(1) > minObjHeight
            cropped = imcrop(im, [currBox(2), currBox(1), currBox(4)-currBox(2), currBox(3)-currBox(1)]);
            feature = extractHOGFeatures(imresize(cropped, imageSize), 'CellSize', cellSize);
            
            % Check false car
            [predictLabel, score] = predict(SVMModelCar, feature);
            if predictLabel(1) == 1
                isOverlap = false;
                for o = 1:numel(objects)
                    if strcmp(objects(o).type, 'Car') || strcmp(objects(o).type, 'Van')
                        ratio = rectOverlap(currBox(1), currBox(2), currBox(3), currBox(4), ...
                            objects(o).y1, objects(o).x1, objects(o).y2, objects(o).x2);
                        if ratio > carTh
                            isOverlap = true;
                            break;
                        end
                    end
                end
                if ~isOverlap
                    % Not a car. So this is a FP.
                    imwrite(cropped, sprintf('%s/%06d.png', fpDir, fpIndex));
                    fpIndex = fpIndex + 1;
                    counter = counter + 1;
                end
            end
            
            % Check false pedestrian
            [predictLabel, score] = predict(SVMModelPed, feature);
            if predictLabel(1) == 1
                isOverlap = false;
                for o = 1:numel(objects)
                    if strcmp(objects(o).type, 'Pedestrian') || strcmp(objects(o).type, 'Person_sitting')
                        ratio = rectOverlap(currBox(1), currBox(2), currBox(3), currBox(4), ...
                            objects(o).y1, objects(o).x1, objects(o).y2, objects(o).x2);
                        if ratio > pedTh
                            isOverlap = true;
                            break;
                        end
                    end
                end
                if ~isOverlap
                    % Not a pedestrian. So this is a FP.
                    imwrite(cropped, sprintf('%s/%06d.png', fpDir, fpIndex));
                    fpIndex = fpIndex + 1;
                    counter = counter + 1;
                end
            end
        end
        % Found enough case for this image
        if counter >= fpNum
            break;
        end
    end
end
fprintf('Found %d false positives. Use %f seconds.\n', fpIndex, toc);