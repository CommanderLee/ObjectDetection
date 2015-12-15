clear all; close all;
clc;

training_num = 10;
num_cars = 0;
num_pedestrian = 0;

num_cars_occ = 0;
num_pedestrian_occ = 0;

image_dir = 'E:/Code/ObjectDetection/new_data/train/image';
label_dir = 'E:/Code/ObjectDetection/new_data/train/label';

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

training_num = nimages;
%img_idx=100;


%% show the img
save_car = 'E:/Code/ObjectDetection/crop/cars';
save_ped = 'E:/Code/ObjectDetection/crop/pedestrians';

for num = 1:training_num
% load labels
img_idx = num - 1;
objects = readLabels(label_dir,img_idx);
if mod(num, 100) == 0
    fprintf('%d\n', num);
end

    for obj_idx=1:numel(objects)

        % get all cars
        if(strcmp(objects(obj_idx).type,'Car') && objects(obj_idx).occlusion <= 1)
            %drawBox2D(h,objects(obj_idx));
            img_crop=crop_object(image_dir,img_idx,[objects(obj_idx).x1,objects(obj_idx).y1,objects(obj_idx).x2,objects(obj_idx).y2]);
            %figure,imshow(img_crop);
            %im=imresize(img_crop,[256 256],'lanczos3');
            im = img_crop;
            imwrite(im,sprintf('%s/%06d.png',save_car,num_cars));
            num_cars = num_cars+1;
        elseif(strcmp(objects(obj_idx).type,'Pedestrian') && objects(obj_idx).occlusion <= 1)
            %drawBox2D(h,objects(obj_idx));
            img_crop=crop_object(image_dir,img_idx,[objects(obj_idx).x1,objects(obj_idx).y1,objects(obj_idx).x2,objects(obj_idx).y2]);
            %figure,imshow(img_crop);
            %im=imresize(img_crop,[256 256],'lanczos3');
            im = img_crop;
            imwrite(im,sprintf('%s/%06d.png',save_ped,num_pedestrian));
            num_pedestrian = num_pedestrian+1;
        end
    end
end