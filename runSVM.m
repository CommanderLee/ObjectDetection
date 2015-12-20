% Apply HoG + SVM
%% clear and close everything
clear all; 
close all;

%% Initialization
carDir = 'crop/cars';
carNum = length(dir(fullfile(carDir, '*.png')));

pedDir = 'crop/pedestrians';
pedNum = length(dir(fullfile(pedDir, '*.png')));

bgDir = 'crop/background';
bgNum = length(dir(fullfile(bgDir, '*.png')));

fpDir = 'crop/false_positive';
fpNum = length(dir(fullfile(fpDir, '*.png')));

fprintf('Found %d cars, %d pedestrians, %d background images, %d false positives.\n', ...
    carNum, pedNum, bgNum, fpNum);

%% Load images. 161-175 sec.
nSize = 64;
tic;
rng(1);

carImg = zeros(nSize, nSize, carNum);
counter = 1;
for i = 1:carNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', carDir, i-1)));
    if size(img, 1) > 30
        im1 = imresize(img, [nSize nSize]);
        im2 = fliplr(im1);
        carImg(:,:,counter) = im1;
        carImg(:,:,counter+1) = im2;
        counter = counter + 2;
    end
end
carNum = counter-1;
carImg = carImg(:,:, 1:carNum);
fprintf('    %d selected cars(augmented).\n', carNum);

pedImg = zeros(nSize, nSize, pedNum);
counter = 1;
for i = 1:pedNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', pedDir, i-1)));
    if size(img, 1) > 30
        im1 = imresize(img, [nSize nSize]);
        im2 = imadjust(im1);
        randBias = rand() * 0.4 - 0.2;
        im3 = imadjust(im1, [0;1], [max(0, randBias);min(1, 1+randBias)]);
        randBias = rand() * 0.4 - 0.2;
        im4 = imadjust(im1, [0;1], [max(0, randBias);min(1, 1+randBias)]);
        randBias = rand() * 0.4 - 0.2;
        im5 = imadjust(im1, [0;1], [max(0, randBias);min(1, 1+randBias)]);
        randBias = rand() * 0.4 - 0.2;
        im6 = imadjust(im1, [0;1], [max(0, randBias);min(1, 1+randBias)]);
        pedImg(:,:,counter) = im1;
        pedImg(:,:,counter+1) = im2;
        pedImg(:,:,counter+2) = im3;
        pedImg(:,:,counter+3) = im4;
        pedImg(:,:,counter+4) = im5;
        pedImg(:,:,counter+5) = im6;
        pedImg(:,:,counter+6) = fliplr(im1);
        pedImg(:,:,counter+7) = fliplr(im2);
        counter = counter + 8;
    end
end
pedNum = counter-1;
pedImg = pedImg(:,:, 1:pedNum);
fprintf('    %d selected pedestrians(augmented).\n', pedNum);

tempBgNum = min(20000, bgNum);
tempBgIndex = randperm(bgNum, tempBgNum)-1;
bgImg = zeros(nSize, nSize, tempBgNum + fpNum);
for i = 1:tempBgNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', bgDir, tempBgIndex(i))));
    bgImg(:,:,i) = imresize(img, [nSize nSize]);
end
for j = 1:fpNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', fpDir, j-1)));
    bgImg(:,:,j + tempBgNum) = imresize(img, [nSize nSize]);
end
bgNum = tempBgNum + fpNum
fprintf('    %d background images.\n', bgNum);

fprintf('Load Images: %f seconds\n', toc);

%% Extract HoG for training & testing set. 65-70 sec.
tic;
cellSize = [8 8];
% img = rgb2gray(imread(sprintf('%s/%06d.png', carDir, 0)));
% img = imresize(img, [nSize nSize]);
img = carImg(:,:,1);
[hogFeature, hogVis] = extractHOGFeatures(img, 'CellSize', cellSize);
% figure;
% imshow(img); 
% hold on;
% plot(hogVis);
hogFeatureSize = length(hogFeature);

% randCarIndex = randperm(carNum, pedNum);
% carNum = pedNum;
randCarIndex = randperm(carNum);
randPedIndex = randperm(pedNum);
randBgIndex = randperm(bgNum);
% Keep some of the backgrounds
% tempBg = min(5237, bgNum);
% randBgIndex = randperm(bgNum, tempBg);
% bgNum = tempBg;

ratioTrainTest = 1;
trainCarNum = round(carNum * ratioTrainTest);
trainPedNum = round(pedNum * ratioTrainTest);
trainBgNum = round(bgNum * ratioTrainTest);
testCarNum = carNum - trainCarNum;
testPedNum = pedNum - trainPedNum;
testBgNum = bgNum - trainBgNum;

fprintf('Train&Validate on %d/%d cars, %d/%d pedestrians, %d/%d background images.\n', ...
    trainCarNum, carNum, trainPedNum, pedNum, trainBgNum, bgNum);

% trainNum = trainCarNum+trainPedNum;
trainNum = trainCarNum + trainPedNum + trainBgNum;
trainFeatures = zeros(trainNum, hogFeatureSize, 'single');
trainLabelsCar = zeros(trainNum, 1, 'uint8');
trainLabelsCar(1 : trainCarNum, 1) = 1;
trainLabelsPed = zeros(trainNum, 1, 'uint8');
trainLabelsPed(trainCarNum+1 : trainCarNum+trainPedNum, 1) = 1;

% testNum = carNum + pedNum - trainNum;
testNum = testCarNum + testPedNum + testBgNum;

testFeatures = zeros(testNum, hogFeatureSize, 'single');
testLabelsCar = zeros(testNum, 1, 'uint8');
testLabelsCar(1 : testCarNum, 1) = 1;
testLabelsPed = zeros(testNum, 1, 'uint8');
testLabelsPed(testCarNum+1 : testCarNum+testPedNum, 1) = 1;

for i=1:trainCarNum
    trainFeatures(i, :) = extractHOGFeatures(carImg(:,:,randCarIndex(i)), 'CellSize', cellSize);
end
for j=1:trainPedNum
    i = j + trainCarNum;
    trainFeatures(i, :) = extractHOGFeatures(pedImg(:,:,randPedIndex(j)), 'CellSize', cellSize);
end
for k=1:trainBgNum
    i = k + trainCarNum + trainPedNum;
    trainFeatures(i, :) = extractHOGFeatures(bgImg(:,:,randBgIndex(k)), 'CellSize', cellSize);
end

for i=trainCarNum+1 : carNum
    testFeatures(i - trainCarNum, :) = extractHOGFeatures(carImg(:,:,randCarIndex(i)), 'CellSize', cellSize);
end
dj = testCarNum - trainPedNum;
for j=trainPedNum+1 : pedNum
    i = j + dj;
    testFeatures(i, :) = extractHOGFeatures(pedImg(:,:,randPedIndex(j)), 'CellSize', cellSize);
end
dk = testCarNum + testPedNum - trainBgNum;
for k=trainBgNum+1 : bgNum
    i = k + dk;
    testFeatures(i, :) = extractHOGFeatures(bgImg(:,:,randBgIndex(k)), 'CellSize', cellSize);
end

clear carImg pedImg bgImg randCarIndex randBgIndex randPedIndex;

fprintf('Extract HoG Features: %f seconds\n', toc);
%% Train, Validate, Test. Save the models.
needValid = false;
kFoldNum = 7;
needTest = false;

%% Train two OvA (One vs All) classifiers. 
%% Car vs all : 2389 + 10664 + ? sec.
fprintf('Preparing SVM: Car vs all.\n');
tic;
SVMModelCar = fitcsvm(trainFeatures, trainLabelsCar);
save('svm_car.mat', 'SVMModelCar');
fprintf('    Train SVM: %f seconds\n', toc);

if needValid
    tic;
    CVSVMModel = crossval(SVMModelCar, 'KFold', kFoldNum);
    classLoss = kfoldLoss(CVSVMModel);
    fprintf('    Cross Validation SVM: Loss:%f, and Time:%f seconds\n', classLoss, toc);
end

if needTest
    tic;
    [predictLabel, score] = predict(SVMModelCar, testFeatures);
    corrVec = (predictLabel == testLabelsCar);
    corrRate = sum(corrVec) / size(predictLabel, 1);
    TP = sum(corrVec(1:testCarNum));
    precision = TP / sum(predictLabel);
    recall = TP / testCarNum;
    fprintf('    Correct rate:%f, Precision:%f, Recall:%f\n', corrRate, precision, recall);
    fprintf('    Test SVM: %f seconds\n', toc);
end

%% Pedestrian vs all : 1207 + 5439 + 44 sec.
fprintf('Preparing SVM: Pedestrian vs all.\n');
tic;
SVMModelPed = fitcsvm(trainFeatures, trainLabelsPed);
save('svm_ped.mat', 'SVMModelPed');
fprintf('    Train SVM: %f seconds\n', toc);

if needValid
    tic;
    CVSVMModel = crossval(SVMModelPed, 'KFold', kFoldNum);
    classLoss = kfoldLoss(CVSVMModel);
    fprintf('    Cross Validation SVM: Loss:%f, and Time:%f seconds\n', classLoss, toc);
end

if needTest
    tic;
    [predictLabel, score] = predict(SVMModelPed, testFeatures);
    corrVec = (predictLabel == testLabelsPed);
    corrRate = sum(corrVec) / size(predictLabel, 1);
    TP = sum(corrVec(testCarNum+1 : testCarNum+testPedNum));
    precision = TP / sum(predictLabel);
    recall = TP / testPedNum;
    fprintf('    Correct rate:%f, Precision:%f, Recall:%f\n', corrRate, precision, recall);
    fprintf('    Test SVM: %f seconds\n', toc);
end
%% Car vs Pedestrian : 325 + 1355 + 16 sec.
fprintf('Preparing SVM: Car vs Pedestrian.\n');
tic;
SVMModelCarPed = fitcsvm(trainFeatures(1 : trainCarNum+trainPedNum, :), ...
    trainLabelsCar(1 : trainCarNum+trainPedNum));
save('svm_car_ped.mat', 'SVMModelCarPed');
fprintf('    Train SVM: %f seconds\n', toc);

if needValid
    tic;
    CVSVMModel = crossval(SVMModelCarPed, 'KFold', kFoldNum);
    classLoss = kfoldLoss(CVSVMModel);
    fprintf('    Cross Validation SVM: Loss:%f, and Time:%f seconds\n', classLoss, toc);
end

if needTest
    tic;
    [predictLabel, score] = predict(SVMModelCarPed, testFeatures(1 : testCarNum+testPedNum, :));
    corrVec = (predictLabel == testLabelsCar(1 : testCarNum+testPedNum));
    corrRate = sum(corrVec) / size(predictLabel, 1);
    TP = sum(corrVec(1:testCarNum));
    precision = TP / sum(predictLabel);
    recall = TP / testCarNum;
    fprintf('    Correct rate:%f, Precision:%f, Recall:%f\n', corrRate, precision, recall);
    fprintf('    Test SVM: %f seconds\n', toc);
end

clear trainFeatures testFeatures SVMModelCar SVMModelPed SVMModelCarPed;