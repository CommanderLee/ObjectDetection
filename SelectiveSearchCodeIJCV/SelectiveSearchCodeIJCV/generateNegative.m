% Generate the negative examples for the object detection task
% Modified from demo.m - Zhen
% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
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

%% Load images
imageDir = 'E:/Code/ObjectDetection/new_data/train/image';
imageNum = length(dir(fullfile(imageDir, '*.png')));
labelDir = 'E:/Code/ObjectDetection/new_data/train/label';
labelNum = length(dir(fullfile(labelDir, '*.txt')));

assert(labelNum == imageNum);
fprintf('Found %d images(labels).\n', labelNum);

imageNum = 10;%For debug

%%
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

backgroundDir = 'E:/Code/ObjectDetection/crop/background';
carTh = 0.5;
pedTh = 0.3;
minBgHeight = 30;
bgNum = 2;
bgIndex = 0;
rng(1);
tic;
fprintf('Looking for background examples.\n');
% For each image, choose 'bgNum' background:
for i=1:imageNum
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    
    % As an example, use a single image
    im = imread(sprintf('%s/%06d.png', imageDir, i-1));

    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);

    % Read labels
    objects = readLabels(labelDir, i);
    
    % Calculate overlap and save negative cases
    boxNum = size(boxes, 1);
    randBoxIndex = randperm(boxNum);
    counter = 0;
    % For each bounding box, check overlap with positive case
    for b = 1:boxNum
        currBox = boxes(randBoxIndex(b), :);
        if currBox(3) - currBox(1) > minBgHeight
            isOverlap = false;
            % Check each positive case
            for o = 1:numel(objects)
                if strcmp(objects(o).type, 'Car') || strcmp(objects(o).type, 'Van')
                    ratio = rectOverlap(currBox(1), currBox(2), currBox(3), currBox(4), ...
                        objects(o).y1, objects(o).x1, objects(o).y2, objects(o).x2);
                    if ratio > carTh
                        isOverlap = true;
                        break;
                    end
                elseif strcmp(objects(o).type, 'Pedestrian') || strcmp(objects(o).type, 'Person_sitting')
                    ratio = rectOverlap(currBox(1), currBox(2), currBox(3), currBox(4), ...
                        objects(o).y1, objects(o).x1, objects(o).y2, objects(o).x2);
                    if ratio > pedTh
                        isOverlap = true;
                        break;
                    end
                end
            end
            % If not overlapped with positive case
            if ~isOverlap
               imwrite(im(currBox(1):currBox(3), currBox(2):currBox(4)), ...
                   sprintf('%s/%06d.png', backgroundDir, bgIndex));
               bgIndex = bgIndex + 1;
               counter = counter + 1;
               % Found enough negative case for this image
               if counter >= bgNum
                   break;
               end
            end
        end
    end
    
    % Show boxes
%     ShowRectsWithinImage(boxes, 5, 5, im);

    % Show blobs which result from first similarity function
%     hBlobs = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{1});
%     ShowBlobs(hBlobs, 5, 5, im);
end
fprintf('Found %d background examples. %f seconds.\n', bgIndex, toc);