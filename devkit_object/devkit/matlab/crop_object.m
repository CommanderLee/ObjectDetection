function [ img_crop ] = crop_object( image_dir, image_idx,coordination)
%crop object img
%
img = imread(sprintf('%s/%06d.png',image_dir,image_idx));
rec=[coordination(1),coordination(2),coordination(3)-coordination(1),coordination(4)-coordination(2)];
img_crop = imcrop(img,rec);

end

