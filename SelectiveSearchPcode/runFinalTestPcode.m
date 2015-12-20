% Run final test
% Using Selective Search -> Test with trained SVM -> Output results.
% Modified from demo.m - Zhen
% This demo shows how to use the software described in our ICCV paper:
%   Segmentation as Selective Search for Object Recognition,
%   K.E.A. van de Sande, J.R.R. Uijlings, T. Gevers, A.W.M. Smeulders, ICCV 2011
%% Initialization
clear all;
close all;

fprintf('Demo of how to run the code for:\n');
fprintf('   K. van de Sande, J. Uijlings, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   ICCV 2011\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex anigaussm/anigauss_mex.c anigaussm/anigauss.c -output anigauss
end


% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
    %     fprintf('
    mex FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%% Load images & labels from the test set
imageDir = 'E:/Code/ObjectDetection/new_data/test/image';
imageNum = length(dir(fullfile(imageDir, '*.png')));
% labelDir = 'E:/Code/ObjectDetection/new_data/test/label';
% labelNum = length(dir(fullfile(labelDir, '*.txt')));
outputDir = 'E:/Code/ObjectDetection/new_data/test/output';

fprintf('Found %d images.\n', imageNum);

imageNum = 1;%For debug

%% Set parameters & Load classifiers
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Rgb', 'Hsv', 'RGI', 'Opp'};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
kThresholds = [200];
sigma = 0.8;
numHierarchy = length(colorTypes) * length(kThresholds);

% Minium bounding box height that we consider.
% Ref: Readme file from development kit
minObjHeight = 25;

rng(1);

load('E:/Code/ObjectDetection/svmModel/svmFinal/svm_car.mat');
load('E:/Code/ObjectDetection/svmModel/svmFinal/svm_ped.mat');
load('E:/Code/ObjectDetection/svmModel/svmFinal/svm_car_ped.mat');
cellSize = [8 8];
imageSize = [64 64];

%% Start testing
tic;
fprintf('Start testing.\n');

boxes = cell(1, imageNum);
for i=1:imageNum
    idx = 1;
    currBox = cell(1, numHierarchy);
    im = imread(sprintf('%s/%06d.png', imageDir, i-1));
    
    for k = kThresholds
        minSize = k; % We use minSize = k.
        
        for colorTypeI = 1:length(colorTypes)
            colorType = colorTypes{colorTypeI};
            
            currBox{idx} = SelectiveSearch(im, sigma, k, minSize, colorType);
            idx = idx + 1;
        end
    end
    
    boxes{i} = cat(1, currBox{:}); % Concatenate results of all hierarchies
    boxes{i} = unique(boxes{i}, 'rows'); % Remove duplicate boxes
    
    boxNum = numel(boxes{i});
    objIndex = 1;
    objects = [];
    carNum = 0;
    pedNum = 0;
    % For each bounding box, test each box with SVM
    for b = 1:boxNum
        currBox = boxes{i}(b, :);
        if currBox(3) - currBox(1) > minObjHeight
            cropped = imcrop(im, [currBox(2), currBox(1), currBox(4)-currBox(2), currBox(3)-currBox(1)]);
            feature = extractHOGFeatures(imresize(cropped, imageSize), 'CellSize', cellSize);
            
            % Test with classifiers
            [predictLabelCar, scoreCar] = predict(SVMModelCar, feature);
            [predictLabelPed, scorePed] = predict(SVMModelPed, feature);
            
            if predictLabelCar(1) == 1 && predictLabelPed(1) == 1
                %                 [predictLabel, score] = predict(SVMModelCarPed, feature);
                %                 if predictLabel(1) == 1
                %                     objects(objIndex).type  = 'Car';
                %                     carNum = carNum + 1;
                %                 else
                %                     objects(objIndex).type  = 'Pedestrian';
                %                     pedNum = pedNum + 1;
                %                 end
                %                 objects(objIndex).score = score(1);
                if scoreCar(1) > scorePed(1)
                    objects(objIndex).type  = 'Car';
                    carNum = carNum + 1;
                    objects(objIndex).score = scoreCar(1);
                else
                    objects(objIndex).type  = 'Pedestrian';
                    pedNum = pedNum + 1;
                    objects(objIndex).score = scorePed(1);
                end
                
                
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
fprintf('Elapsed time: %f seconds\n', toc);

%% Show a couple of good boxes in the image
fprintf('Showing examples of good boxes\n');
goodBoxes = boxes{1}([48 1075 808 762 467 445], :);
%goodBoxes = boxes{1}([1 4 54 4 43 211], :);
figure;
for i=1:6
    subplot(2, 3, i);
    boxIm = im(goodBoxes(i,1):goodBoxes(i,3), goodBoxes(i,2):goodBoxes(i,4), :);
    imshow(boxIm);
end

%%
% Test overlap scores Pascal 2007 test
if exist('SelectiveSearchVOC2007test.mat')
    load GroundTruthVOC2007test.mat; % Load ground truth boxes
    load SelectiveSearchVOC2007test.mat; % Load selective search boxes
    
    % Remove small boxes
    for i=1:length(boxes)
        [nR nC] = BoxSize(boxes{i});
        keepIdx = min(nR, nC) > 20;% Keep boxes with width/height > 20 pixels
        boxes{i} = boxes{i}(keepIdx,:);
        numberBoxes(i) = size(boxes{i},1);
    end
    
    % Get for each ground truth box the best Pascal Overlap Score
    maxScores = MaxOverlapScores(gtBoxes, gtImIds, boxes);
    
    % Get recall per class
    for cI=1:length(maxScores)
        recall(cI) = sum(maxScores{cI} > 0.5) ./ length(maxScores{cI});
        averageBestOverlap(cI) = mean(maxScores{cI});
    end
    
    recall
    fprintf('Number of boxes per image: %.0f\nMean Average Best Overlap: %f\nMean Recall: %f\n', mean(numberBoxes), mean(averageBestOverlap), mean(recall));
end

%% Example of segmentation
% sigma = 0.8
% k = 100
% minSize = 200
% segIndIm = mexFelzenSegmentIndex(im, 0.8, 100, 200);

% segIndIm has the same number of rows and columns as im. The range of
% segIndIm is 1:S, where S is the number of segments: Each number
% in segIndIm corresponds to the segment the pixel belongs to.
