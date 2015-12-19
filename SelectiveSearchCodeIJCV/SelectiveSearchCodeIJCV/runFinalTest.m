% Run final test
% Using Selective Search -> Test with trained SVM -> Output results.
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

%% Load images & labels from the test set
imageDir = 'E:/Code/ObjectDetection/new_data/test/image';
imageNum = length(dir(fullfile(imageDir, '*.png')));
% labelDir = 'E:/Code/ObjectDetection/new_data/test/label';
% labelNum = length(dir(fullfile(labelDir, '*.txt')));
outputDir = 'E:/Code/ObjectDetection/new_data/test/output';

assert(labelNum == imageNum);
fprintf('Found %d images(labels).\n', labelNum);

imageNum = 1;%For debug

%% Set parameters & Load classifiers
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
% Minium bounding box height that we consider. 
% Ref: Readme file from development kit
minObjHeight = 25;

rng(1);

load('svm_car.mat');
load('svm_ped.mat');
load('svm_car_ped.mat');
cellSize = [8 8];
imageSize = [64 64];

%% Start testing
tic;
fprintf('Start testing.\n');

for i=1:imageNum
    im = imread(sprintf('%s/%06d.png', imageDir, i-1));

    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    boxNum = size(boxes, 1);
    
    objIndex = 1;
    objects = [];
    carNum = 0;
    pedNum = 0;
    % For each bounding box, test each box with SVM
    for b = 1:boxNum
        currBox = boxes(randBoxIndex(b), :);
        if currBox(3) - currBox(1) > minObjHeight
            cropped = imcrop(im, [currBox(2), currBox(1), currBox(4)-currBox(2), currBox(3)-currBox(1)]);
            feature = extractHOGFeatures(imresize(cropped, imageSize), 'CellSize', cellSize);
            
            % Test with classifiers
            [predictLabelCar, scoreCar] = predict(SVMModelCar, feature);
            [predictLabelPed, scorePed] = predict(SVMModelPed, feature);

            if predictLabelCar(1) == 1 && predictLabelPed(1) == 1
                [predictLabel, score] = predict(SVMModelCarPed, feature);
                if predictLabel(1) == 1
                    objects(objIndex).type  = 'Car';
                    carNum = carNum + 1;
                else
                    objects(objIndex).type  = 'Pedestrian';
                    pedNum = pedNum + 1;
                end
                objects(objIndex).score = score(1);
                
            elseif predictLabelCar(1) == 1 && predictLabelPed(1) == 0
                objects(objIndex).type  = 'Car';
                objects(objIndex).score = scoreCar(1);
                carNum = carNum + 1;
                
            elseif predictLabelCar(1) == 0 && predictLabelPed(1) == 1
                objects(objIndex).type  = 'Pedestrian';
                objects(objIndex).score = scorePed(1);
                pedNum = pedNum + 1;
            end
            
            % Find something
            if predictLabelCar(1) + predictLabelPed(1) >= 1
                objects(objIndex).x1    = currBox(2);
                objects(objIndex).y1    = currBox(1);
                objects(objIndex).x2    = currBox(4);
                objects(objIndex).y2    = currBox(3);
                objects(objIndex).alpha = pi/2;
                objIndex = objIndex + 1;
            end
        end
    end
    % write objects to file
    writeLabels(objects, outputDir, i-1);
    fprintf('%06d: Find %d cars and %d pedestrians.\n', i-1, carNum, pedNum);
end
fprintf('Complete! Used %f seconds.\n', toc);