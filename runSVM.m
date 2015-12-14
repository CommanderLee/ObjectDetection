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

%% Load images. 35sec.
nSize = 64;

carImg = zeros(nSize, nSize, carNum);
counter = 0;
tic;
for i = 1:carNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', carDir, i-1)));
    if size(img, 1) > 35
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

%% Extract HoG for training & testing set. 40 sec.
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

rng(1);
% randCarIndex = randperm(carNum, pedNum);
% carNum = pedNum;
randCarIndex = randperm(carNum);
randPedIndex = randperm(pedNum);

trainCarNum = round(carNum * 0.7);
trainPedNum = round(pedNum * 0.7);
fprintf('Train&Validate on %d/%d cars, %d/%d pedestrians.\n', ...
    trainCarNum, carNum, trainPedNum, pedNum);

trainNum = trainCarNum+trainPedNum;
trainFeatures = zeros(trainNum, hogFeatureSize, 'single');
trainLabels = zeros(trainNum, 1, 'uint8');
trainLabels(1:trainCarNum, 1) = 1;

testNum = carNum + pedNum - trainNum;
testFeatures = zeros(testNum, hogFeatureSize, 'single');
testLabels = zeros(testNum, 1, 'uint8');
testLabels(1 : carNum-trainCarNum, 1) = 1;

for i=1:trainCarNum
    trainFeatures(i, :) = extractHOGFeatures(carImg(:,:,randCarIndex(i)), 'CellSize', cellSize);
end
for j=1:trainPedNum
    i = j + trainCarNum;
    trainFeatures(i, :) = extractHOGFeatures(pedImg(:,:,randPedIndex(j)), 'CellSize', cellSize);
end

for i=trainCarNum+1 : carNum
    testFeatures(i - trainCarNum, :) = extractHOGFeatures(carImg(:,:,randCarIndex(i)), 'CellSize', cellSize);
end
dj = carNum - trainCarNum - trainPedNum;
for j=trainPedNum+1 : pedNum
    i = j + dj;
    testFeatures(i, :) = extractHOGFeatures(pedImg(:,:,randPedIndex(j)), 'CellSize', cellSize);
end
fprintf('Extract HoG Features: %f seconds\n', toc);
%% Train a classifier. 18 + 117 + 15 sec.
tic;
SVMModel = fitcsvm(trainFeatures, trainLabels);
fprintf('Train SVM: %f seconds\n', toc);

tic;
CVSVMModel = crossval(SVMModel, 'KFold', 7);
classLoss = kfoldLoss(CVSVMModel)
save('svm.mat', 'SVMModel');
fprintf('Cross Validation SVM: %f seconds\n', toc);

tic;
[predictLabel, score] = predict(SVMModel, testFeatures);
errorRes = find(predictLabel ~= testLabels);
errorRate = numel(errorRes) / size(predictLabel, 1);
fprintf('Error rate:%f\n', errorRate);
fprintf('Test SVM: %f seconds\n', toc);