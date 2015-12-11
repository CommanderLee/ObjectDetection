function runCNN
%% clear and close everything
clear all; 
close all;

%% Initialization
rootDir  = '';
imageDir = fullfile(rootDir,'data_object_image_2/training/image_2');
labelDir = fullfile(rootDir,'data_object_image_2/training/label_2');

imageNum = length(dir(fullfile(imageDir, '*.png')));
trainNum = floor(imageNum / 10) * 6;
validNum = floor(imageNum / 10) * 1;
testNum = imageNum - trainNum - validNum;

fprintf('N=%d, train:%d, valid:%d, test:%d.\n', imageNum, trainNum, validNum, testNum);

% Always to refer to randOrder as index, to get the random order of image
randOrder = randperm(imageNum)-1;
end