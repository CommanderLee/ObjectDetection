% countObjects
%% 
close all;
clear all;
%labelDir = strrep(uigetdir(pwd), '\\', '/');
labelDir = 'E:/Code/ObjectDetection/data_object_label_2/training/label_2';
imageMaxNum = 7480;

cars = 0;
vans = 0;
sitting = 0;
pedestrians = 0;

%%
carRatio = [];
pedRatio = [];
for i=0:imageMaxNum
   objects = readLabels(labelDir, i);
   for o = 1:numel(objects)
       %fprintf('%s\n', objects(o).type);
       if strcmp(objects(o).type, 'Car')
           cars = cars + 1;
           carRatio = [carRatio, (objects(o).x2-objects(o).x1)/(objects(o).y2-objects(o).y1)];
       elseif  strcmp(objects(o).type, 'Van')
           vans = vans + 1;
       elseif strcmp(objects(o).type, 'Pedestrian')
           pedestrians = pedestrians + 1;
           pedRatio = [pedRatio, (objects(o).x2-objects(o).x1)/(objects(o).y2-objects(o).y1)];
       elseif strcmp(objects(o).type, 'Person_sitting')
           sitting = sitting + 1;
       end
   end
end

fprintf('Car:%d, Pedestrian:%d, Van:%d, Sitting:%d\n', cars, pedestrians, vans, sitting);

%% Analyze Aspect Ratio: 
% CarRatio: mean:1.720363 sigma:0.600885, PedestrianRatio: mean:0.401079 sigma:0.111491.
fprintf('CarRatio: mean:%f sigma:%f, PedestrianRatio: mean:%f sigma:%f.\n', ...
    mean(carRatio), sqrt(var(carRatio)), mean(pedRatio), sqrt(var(pedRatio)));
x = 0:0.2:5;
subplot(1, 2, 1);
hCar = hist(carRatio, x);
newValuesC = hCar / sum(hCar);
bar(x, newValuesC, 'style', 'histc');
xlim([0 4.4]);
ylim([0 0.8]);
title('Cars');

subplot(1, 2, 2);
hPed = hist(pedRatio, x);
newValuesP = hPed / sum(hPed);
bar(x, newValuesP, 'style', 'histc');
xlim([0 4.4]);
ylim([0 0.8]);
title('Pedestrians');