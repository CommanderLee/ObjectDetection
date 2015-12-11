clear all; close all;
clc;

disp('======= KITTI Object Detection Regionlets =======');

% options
root_dir = '../../../../data_object_image';
data_set = 'training';

training_num = 10;
num_cars = 0;
num_pedestrian = 0;

num_cars_occ = 0;
num_pedestrian_occ = 0;

% get sub-directories
cam = 2; % 2 = left color camera
%image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
train_type = '/cars';
image_dir = fullfile(root_dir,['/crop' train_type]);
%label_dir = fullfile(root_dir,[data_set '/label_' num2str(cam)]);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));
% set up figure
%h = visualization('init',image_dir);
training_num = nimages;

%% show the img

for num = 1:training_num
% load labels
img_idx = num;
%objects = readLabels(label_dir,img_idx);
  
% visualization update for next frame
%visualization('update',image_dir,h,img_idx,nimages,data_set);
img = imread(sprintf('%s/%06d.png',image_dir,img_idx));
im = im2double(img);
feat = features(im, 8);
ihog = invertHOG(feat);

figure(1);
clf;

subplot(131);
imagesc(im); axis image; axis off;
title('Original Image', 'FontSize', 20);

subplot(132);
showHOG(feat); axis off;
title('HOG Features', 'FontSize', 20);

subplot(133);
imagesc(ihog); axis image; axis off;
title('HOG Inverse', 'FontSize', 20);

    waitforbuttonpress; 
    key = get(gcf,'CurrentCharacter');
   switch lower(key)                         
    case 'q',  break;                                 % quit
    %case '-',  img_idx = max(img_idx-1,  0);          % previous frame
    %case 'x',  img_idx = min(img_idx+1000,nimages-1); % +100 frames
    %case 'y',  img_idx = max(img_idx-1000,0);         % -100 frames
    otherwise, continue;  % next frame
  end

end