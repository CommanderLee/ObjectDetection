function out = countObjects()
%% 
%labelDir = strrep(uigetdir(pwd), '\\', '/');
labelDir = 'E:/Code/ObjectDetection/data_object_label_2/training/label_2';
imageMaxNum = 7480;

cars = 0;
vans = 0;
sitting = 0;
pedestrians = 0;

%%
for i=0:imageMaxNum
   objects = readLabels(labelDir, i);
   for o = 1:numel(objects)
       %fprintf('%s\n', objects(o).type);
       if strcmp(objects(o).type, 'Car')
           cars = cars + 1;
       elseif  strcmp(objects(o).type, 'Van')
           vans = vans + 1;
       elseif strcmp(objects(o).type, 'Pedestrian')
           pedestrians = pedestrians + 1;
           if round(objects(o).x2) - round(objects(o).x1) == 18 && round(objects(o).y2) - round(objects(o).y1) == 214
               fprintf('215*19: %d-%d\n', i, o);
           end
           if round(objects(o).x2) - round(objects(o).x1) == 114 && round(objects(o).y2) - round(objects(o).y1) == 227
               fprintf('228*115: %d-%d\n', i, o);
           end 
       elseif strcmp(objects(o).type, 'Person_sitting')
           sitting = sitting + 1;
       end
   end
end

fprintf('Car:%d, Pedestrian:%d, Van:%d, Sitting:%d\n', cars, pedestrians, vans, sitting);

end