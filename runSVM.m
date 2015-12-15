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

fprintf('Found %d cars, %d pedestrians, %d background images.\n', carNum, pedNum, bgNum);

%% Load images. 95 sec.
nSize = 64;
tic;

carImg = zeros(nSize, nSize, carNum);
counter = 0;
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

pedImg = zeros(nSize, nSize, pedNum);
counter = 0;
for i = 1:pedNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', pedDir, i-1)));
    if size(img, 1) > 30
        counter = counter + 1;
        pedImg(:,:,counter) = imresize(img, [nSize nSize]);
    end
end
pedNum = counter;
pedImg = pedImg(:,:, 1:pedNum);
fprintf('    %d selected pedestrians.\n', pedNum);

bgImg = zeros(nSize, nSize, bgNum);
for i = 1:bgNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', bgDir, i-1)));
    bgImg(:,:,i) = imresize(img, [nSize nSize]);
end
fprintf('    %d background images.\n', bgNum);

fprintf('Load Images: %f seconds\n', toc);

%% Extract HoG for training & testing set. 30 sec.
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
randBgIndex = randperm(bgNum);

trainCarNum = round(carNum * 0.7);
trainPedNum = round(pedNum * 0.7);
trainBgNum = round(bgNum * 0.7);
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

fprintf('Extract HoG Features: %f seconds\n', toc);
%% Train two OvA (One vs All) classifiers. 
% Car vs all : 161 + 1053 + 48 sec.
fprintf('Preparing SVM: Car vs all.\n');
tic;
SVMModelCar = fitcsvm(trainFeatures, trainLabelsCar);
fprintf('    Train SVM: %f seconds\n', toc);

tic;
CVSVMModel = crossval(SVMModelCar, 'KFold', 7);
classLoss = kfoldLoss(CVSVMModel)
save('svm_car.mat', 'SVMModelCar');
fprintf('    Cross Validation SVM: %f seconds\n', toc);

tic;
[predictLabel, score] = predict(SVMModelCar, testFeatures);
corrVec = (predictLabel == testLabelsCar);
corrRate = sum(corrVec) / size(predictLabel, 1);
TP = sum(corrVec(1:testCarNum));
precision = TP / sum(predictLabel);
recall = TP / testCarNum;
fprintf('    Correct rate:%f, Precision:%f, Recall:%f\n', corrRate, precision, recall);
fprintf('    Test SVM: %f seconds\n', toc);

% Pedestrian vs all : 103 + 721 + 32sec.
fprintf('Preparing SVM: Pedestrian vs all.\n');
tic;
SVMModelPed = fitcsvm(trainFeatures, trainLabelsPed);
fprintf('    Train SVM: %f seconds\n', toc);

tic;
CVSVMModel = crossval(SVMModelPed, 'KFold', 7);
classLoss = kfoldLoss(CVSVMModel)
save('svm_ped.mat', 'SVMModelPed');
fprintf('    Cross Validation SVM: %f seconds\n', toc);

tic;
[predictLabel, score] = predict(SVMModelPed, testFeatures);
corrVec = (predictLabel == testLabelsPed);
corrRate = sum(corrVec) / size(predictLabel, 1);
TP = sum(corrVec(testCarNum+1 : testCarNum+testPedNum));
precision = TP / sum(predictLabel);
recall = TP / testPedNum;
fprintf('    Correct rate:%f, Precision:%f, Recall:%f\n', corrRate, precision, recall);
fprintf('    Test SVM: %f seconds\n', toc);

%% Car vs Pedestrian : 27 + 181 + 11 sec.
fprintf('Preparing SVM: Car vs Pedestrian.\n');
tic;
SVMModelCarPed = fitcsvm(trainFeatures(1 : trainCarNum+trainPedNum, :), ...
    trainLabelsCar(1 : trainCarNum+trainPedNum));
fprintf('    Train SVM: %f seconds\n', toc);

tic;
CVSVMModel = crossval(SVMModelCarPed, 'KFold', 7);
classLoss = kfoldLoss(CVSVMModel)
save('svm_car_ped.mat', 'SVMModelCarPed');
fprintf('    Cross Validation SVM: %f seconds\n', toc);

tic;
[predictLabel, score] = predict(SVMModelCarPed, testFeatures(1 : testCarNum+testPedNum, :));
corrVec = (predictLabel == testLabelsCar(1 : testCarNum+testPedNum));
corrRate = sum(corrVec) / size(predictLabel, 1);
TP = sum(corrVec(1:testCarNum));
precision = TP / sum(predictLabel);
recall = TP / testCarNum;
fprintf('    Correct rate:%f, Precision:%f, Recall:%f\n', corrRate, precision, recall);
fprintf('    Test SVM: %f seconds\n', toc);