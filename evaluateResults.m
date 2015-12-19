% Evalulate the output results 
%% Initialization
clear all;
close all;

%% Load images & correct labels from the test set
imageDir = 'E:/Code/ObjectDetection/new_data/test/image';
imageNum = length(dir(fullfile(imageDir, '*.png')));
labelDir = 'E:/Code/ObjectDetection/new_data/test/label';
labelNum = length(dir(fullfile(labelDir, '*.txt')));
outputDir = 'E:/Code/ObjectDetection/new_data/test/output';
outputNum = length(dir(fullfile(outputDir, '*.txt')));

assert(labelNum == imageNum);
assert(labelNum == outputNum);
fprintf('Found %d images(labels).\n', labelNum);

imageNum = 1;%For debug

%% Start evaluation
tic;
fprintf('Start comparing results.\n');

for i=1:imageNum
    if mod(i,100) == 0
        fprintf('%d\n', i);
    end
    
    % Read labels
    correctObjects = readLabels(labelDir, i-1);
    outputObjects = readLabels(outputDir, i-1);
    
    
end