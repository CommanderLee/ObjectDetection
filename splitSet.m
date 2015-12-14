% Split the original training set to 70%+30% sets.
%% Load image & label directory
clear all;
close all;

labelDir = 'data_object_label_2/training/label_2';
labelNum = length(dir(fullfile(labelDir, '*.txt')));
imageDir = 'data_object_image_2/training/image_2';
imageNum = length(dir(fullfile(imageDir, '*.png')));
assert(labelNum == imageNum);
fprintf('Found %d images(labels).\n', labelNum);

%% Generate test set index (30% of the data)
rng(1);
randIndex = randperm(imageNum)-1;
save('randIndex.mat', 'randIndex');

testNum = round(imageNum * 0.3);
trainNum = imageNum - testNum;

%% Split and move to new directory
trainLabelDir = 'new_data/train/label';
trainImageDir = 'new_data/train/image';
testLabelDir = 'new_data/test/label';
testImageDir = 'new_data/test/image';

for i=1:trainNum
    movefile(sprintf('%s/%06d.png', imageDir, randIndex(i)), ...
        sprintf('%s/%06d.png', trainImageDir, i-1));
    movefile(sprintf('%s/%06d.txt', labelDir, randIndex(i)), ...
        sprintf('%s/%06d.txt', trainLabelDir, i-1));
end
for j=1:testNum
    i = j + trainNum;
    movefile(sprintf('%s/%06d.png', imageDir, randIndex(i)), ...
        sprintf('%s/%06d.png', testImageDir, j-1));
    movefile(sprintf('%s/%06d.txt', labelDir, randIndex(i)), ...
        sprintf('%s/%06d.txt', testLabelDir, j-1));
end