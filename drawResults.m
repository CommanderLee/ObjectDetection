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
% assert(labelNum == outputNum);
fprintf('Found %d images(labels).\n', labelNum);

imageNum = 1;%For debug

%% Start evaluation
h = figure();
outObjects = readOutputLabels(outputDir, 15);
drawBox2D(h, object);
%%
tic;
fprintf('Start drawing.\n');

% for i=1:imageNum
for i=16:17
    if mod(i,100) == 0
        fprintf('%d\n', i);
    end
    
    % Read labels
    corrObjects = readLabels(labelDir, i-1);
    corrObjNum = numel(corrObjects);
    outObjects = readOutputLabels(outputDir, i-1);
    outObjNum = numel(outObjects);
    % Increase if we find this object
    findObjects = zeros(corrObjNum, 1);
    
    % For each output objects
    for iOut = 1:outObjNum
        if strcmp(outObjects(iOut).type, 'Car')
            found = false;
            % Check each correct cars and vans
            for iCorr = 1:corrObjNum
                if strcmp(corrObjects(iCorr).type, 'Car') || strcmp(corrObjects(iCorr).type, 'Van')
                    ratio = rectOverlap(outObjects(iOut).y1, outObjects(iOut).x1, outObjects(iOut).y2, outObjects(iOut).x2, ...
                        corrObjects(iCorr).y1, corrObjects(iCorr).x1, corrObjects(iCorr).y2, corrObjects(iCorr).x2);
                    if ratio > carTh
                        % Found a corresponding object.
                        findObjects(iCorr) = findObjects(iCorr) + 1;
                        found = true;
                    end
                end
            end
            if ~found
                % Claim a 'car' but actually not.
                carFP = carFP + 1;
            end
        elseif strcmp(outObjects(iOut).type, 'Pedestrian')
            found = false;
            % Check each correct pedestrians and sitting persons
            for iCorr = 1:corrObjNum
                if strcmp(corrObjects(iCorr).type, 'Pedestrian') || strcmp(corrObjects(iCorr).type, 'Person_sitting')
                    ratio = rectOverlap(outObjects(iOut).y1, outObjects(iOut).x1, outObjects(iOut).y2, outObjects(iOut).x2, ...
                        corrObjects(iCorr).y1, corrObjects(iCorr).x1, corrObjects(iCorr).y2, corrObjects(iCorr).x2);
                    if ratio > pedTh
                        % Found a corresponding object.
                        findObjects(iCorr) = findObjects(iCorr) + 1;
                        found = true;
                    end
                end
            end
            if ~found
                % Claim a 'pedestrian' but actually not.
                pedFP = pedFP + 1;
            end
        else
            fprintf('Error: wrong type name.\n');
        end
    end
    
    % Accumulate TP and FN for this image
    for iCorr = 1:corrObjNum
       if findObjects(iCorr) > 0
           % TP:
           if strcmp(corrObjects(iCorr).type, 'Car') || strcmp(corrObjects(iCorr).type, 'Van')
               carTP = carTP + 1;
           elseif strcmp(corrObjects(iCorr).type, 'Pedestrian') || strcmp(corrObjects(iCorr).type, 'Person_sitting')
               pedTP = pedTP + 1;
           end
       else
           % FN:
           if strcmp(corrObjects(iCorr).type, 'Car')
               carFN = carFN + 1;
           elseif strcmp(corrObjects(iCorr).type, 'Pedestrian')
               pedFN = pedFN + 1;
           end
       end
    end
end
fprintf('Finished. Used %f seconds.\n', toc);

carPrecision = carTP / (carTP + carFP);
carRecall = carTP / (carTP + carFN);
pedPrecision = pedTP / (pedTP + pedFP);
pedRecall = pedTP / (pedTP + pedFN);

Result = sprintf('    & TP & FP & FN & Precision & Recall\\\\ \nCar & %d & %d & %d & %f & %f\nPed & %d & %d & %d & %f & %f\n', ...
    carTP, carFP, carFN, carPrecision, carRecall, ...
    pedTP, pedFP, pedFN, pedPrecision, pedRecall);
fprintf(Result);

% Save to file
fid = fopen('results.txt','w');
fprintf(fid, Result);
fclose(fid);