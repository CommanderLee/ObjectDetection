function runCNN
%% clear and close everything
clear all; 
close all;

%% Initialization
rootDir  = '';
imageDir = fullfile(rootDir,'data_object_image_2/training/image_2');
labelDir = fullfile(rootDir,'data_object_label_2/training/label_2');

imageNum = length(dir(fullfile(imageDir, '*.png')));
labelNum = length(dir(fullfile(labelDir, '*.txt')));
assert(imageNum == labelNum);
% trainNum = floor(imageNum / 10) * 6;
% validNum = floor(imageNum / 10) * 1;
% testNum = imageNum - trainNum - validNum;
% 
% fprintf('N=%d, train:%d, valid:%d, test:%d.\n', imageNum, trainNum, validNum, testNum);

%% Count objects
cars = 0;
vans = 0;
sitting = 0;
pedestrians = 0;
others = 0;

for i=0:(labelNum-1)
   objects = readLabels(labelDir, i);
   for o = 1:numel(objects)
       %fprintf('%s\n', objects(o).type);
       if strcmp(objects(o).type, 'Car')
           cars = cars + 1;
       elseif  strcmp(objects(o).type, 'Van')
           vans = vans + 1;
       elseif strcmp(objects(o).type, 'Pedestrian')
           pedestrians = pedestrians + 1;
       elseif strcmp(objects(o).type, 'Person_sitting')
           sitting = sitting + 1;
       elseif ~strcmp(objects(o).type, 'DontCare')
           others = others + 1;
       end
   end
end

fprintf('Car:%d, Pedestrian:%d, Van:%d, Sitting:%d Others:%d\n', ...
cars, pedestrians, vans, sitting, others);

%% 
% Always to refer to randOrder as index, to get the random order of image
randOrder = randperm(imageNum)-1;
for ii = 1:imageNum
    % i is the randomized order, range from [0,num)
    i = randOrder(ii);
end

end