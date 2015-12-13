% Apply HoG + SVM
%% clear and close everything
clear all; 
close all;

%% Initialization
carDir = 'crop/cars';
carNum = length(dir(fullfile(carDir, '*.png')));

pedDir = 'crop/pedestrian';
pedNum = length(dir(fullfile(pedDir, '*.png')));
pedDir1 = 'crop/pedestrian_1';
pedNum1 = length(dir(fullfile(pedDir1, '*.png')));

fprintf('Found %d cars, %d pedestrians.\n', carNum, pedNum+pedNum1);

%% Load images
nSize = 64;

carImg = zeros(nSize, nSize, carNum);
counter = 0;
tic;
for i = 1:carNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', carDir, i-1)));
    if size(img, 1) > 65
        counter = counter + 1;
        carImg(:,:,counter) = imresize(img, [nSize nSize]);
    end
end
carNum = counter;
carImg = carImg(:,:, 1:carNum);
fprintf('    %d selected cars.\n', carNum);

pedImg = zeros(nSize, nSize, pedNum+pedNum1);
counter = 0;
for i = 1:pedNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', pedDir, i-1)));
    if size(img, 1) > 35
        counter = counter + 1;
        pedImg(:,:,counter) = imresize(img, [nSize nSize]);
    end
end
for i = 1:pedNum1
    img = rgb2gray(imread(sprintf('%s/%06d.png', pedDir1, i-1)));
    if size(img, 1) > 35
        counter = counter + 1;
        pedImg(:,:,counter) = imresize(img, [nSize nSize]);
    end
end
pedNum = counter;
pedImg = pedImg(:,:, 1:pedNum);
fprintf('    %d selected pedestrians.\n', pedNum);
fprintf('Load Images: %f seconds\n', toc);

%% Extract HoG
tic;
cellSize = [4 4];
% img = rgb2gray(imread(sprintf('%s/%06d.png', carDir, 0)));
% img = imresize(img, [nSize nSize]);
img = carImg(:,:,1);
[hogFeature, hogVis] = extractHOGFeatures(img, 'CellSize', cellSize);
% figure;
% imshow(img); 
% hold on;
% plot(hogVis);
hogFeatureSize = length(hogFeature);

trainFeatures = zeros(carNum+pedNum, hogFeatureSize, 'single');
trainLabels = zeros(carNum+pedNum, 1, 'uint8');
trainLabels(1:carNum, 1) = 1;

for i=1:carNum
    trainFeatures(i, :) = extractHOGFeatures(carImg(:,:,i), 'CellSize', cellSize);
end

for j=1:pedNum
    i = j + carNum;
    trainFeatures(i, :) = extractHOGFeatures(pedImg(:,:,j), 'CellSize', cellSize);
end

fprintf('Extract HoG Features: %f seconds\n', toc);
%% Train a classifier
tic;
SVMModel = fitcsvm(trainFeatures, trainLabels);
CVSVMModel = crossval(SVMModel, 'KFold', 7);
classLoss = kfoldLoss(CVSVMModel)

fprintf('Train & Validate SVM: %f seconds\n', toc);