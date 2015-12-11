clear all; close all;
clc;

disp('======= KITTI Object Detection Regionlets =======');

% options
root_dir = '../../data_object_image';
data_set = 'training';

training_num = 10;
num_cars = 0;


% get sub-directories
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
label_dir = fullfile(root_dir,[data_set '/label_' num2str(cam)]);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));
% set up figure
h = visualization('init',image_dir);

%img_idx=100;


%% show the img

for num = 1:training_num
% load labels
img_idx = num;
objects = readLabels(label_dir,img_idx);
  
% visualization update for next frame
visualization('update',image_dir,h,img_idx,nimages,data_set);

for obj_idx=1:numel(objects)
   
    % get all cars
    %if(strcmp(objects(obj_idx).type,'Pedestrian'))
    if(strcmp(objects(obj_idx).type,'Car'))
        drawBox2D(h,objects(obj_idx));
        img_crop=crop_object(image_dir,img_idx,[objects(obj_idx).x1,objects(obj_idx).y1,objects(obj_idx).x2,objects(obj_idx).y2]);
        figure,imshow(img_crop);
        im=imresize(img_crop,[256 256],'lanczos3');
        num_cars = num_cars+1;
    end
    
end
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


