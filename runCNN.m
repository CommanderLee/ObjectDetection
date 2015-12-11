function runCNN
%% clear and close everything
clear all; 
close all;

%% Initialization
carDir = 'crop/cars';
carNum = length(dir(fullfile(carDir, '*.png')));
pedDir = 'crop/pedestrian';
pedNum = length(dir(fullfile(pedDir, '*.png')));
othDir = 'crop/others';
othNum = length(dir(fullfile(othDir, '*.png')));

fprintf('Found %d cars, %d pedestrians, and %d other things.\n', carNum, pedNum, othNum);

%% Load images
nSize = 64;

carImg = zeros(nSize, nSize, carNum);
for i = 1:carNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', carDir, i-1)));
    carImg(:,:,i) = imresize(img, [nSize nSize]);
end

pedImg = zeros(nSize, nSize, pedNum);
for i = 1:pedNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', pedDir, i-1)));
    pedImg(:,:,i) = imresize(img, [nSize nSize]);
end

othImg = zeros(nSize, nSize, othNum);
for i = 1:othNum
    img = rgb2gray(imread(sprintf('%s/%06d.png', othDir, i-1)));
    othImg(:,:,i) = imresize(img, [nSize nSize]);
end

%% k-fold validation
K = 7;
errK = [];
carFoldNum = floor(carNum / K);
pedFoldNum = floor(pedNum / K);
othFoldNum = floor(othNum / K);

trainNum = (carFoldNum + pedFoldNum + othFoldNum) * (K-1);
trainX = [];%zeros(nSize, nSize, trainNum);
trainY = zeros(2, trainNum);
trainY(1, 1:carFoldNum*(K-1)) = 1;
trainY(2, carFoldNum*(K-1) + 1 : carFoldNum*(K-1) + pedFoldNum*(K-1)) = 1;
% trainY(3, carFoldNum*(K-1) + pedFoldNum*(K-1) + 1:trainNum) = 1;

validNum = carFoldNum + pedFoldNum + othFoldNum;
validX = [];%zeros(nSize, nSize, validNum);
validY = zeros(2, validNum);
validY(1, 1:carFoldNum) = 1;
validY(2, carFoldNum + 1 : carFoldNum + pedFoldNum) = 1;
% validY(3, carFoldNum + pedFoldNum + 1 : validNum) = 1;

%% Calculate
for k=1:K
    % Use the k-th data as validation
    trainX = cat(3, carImg(:,:, 1:carFoldNum*(k-1)), ...
        carImg(:,:, carFoldNum*k+1 : carFoldNum*K), ...
        pedImg(:,:, 1:pedFoldNum*(k-1)), ...
        pedImg(:,:, pedFoldNum*k+1 : pedFoldNum*K));%, ...
%         othImg(:,:, 1:othFoldNum*(k-1)), ...
%         othImg(:,:, othFoldNum*k+1 : othFoldNum*K)];
    
    validX = cat(3, carImg(:,:, carFoldNum*(k-1)+1 : carFoldNum*k), ...
        pedImg(:,:, pedFoldNum*(k-1)+1 : pedFoldNum*k));%, ...
%         othImg(:,:, othFoldNum*(k-1)+1 : othFoldNum*k)];
    
    %% Train a 6c-2s-12c-2s Convolutional Neural Network 
    rand('state',0)
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
    };
    cnn = cnnsetup(cnn, trainX, trainY);

    opts.alpha = 1;
    opts.batchsize = 50;
    opts.numepochs = 1;

    cutNum = floor(size(trainX, 3) / opts.batchsize) * opts.batchsize;
    cnn = cnntrain(cnn, trainX(:,:,1:cutNum), trainY(:,1:cutNum), opts);
    
    %% Validate and save
    [er, bad] = cnntest(cnn, validX, validY);
    cnnFile = sprintf('cnn_%d.mat', k);
    save(cnnFile, 'cnn');
    fprintf('Save to file: cnn_%d.mat. Error rate: %f\n', k, er);
    errK = [errK; er];
end
disp(errK);

end