function TrainAndSaveEx( MaxIter, splitNum, negLabel )  
%TrainAndSaveEx( MaxIter, splitNum, negLabel )
%   MaxIter: 最大迭代次数,默认200
%   splitNum: 树分叉深度,默认为2
%   negLabel: 负样本标签,默认不指定    

    %默认参数
    isNegLabel = 1;
	if( nargin < 3)
		isNegLabel = -1;
	end
	if( nargin < 2)
		splitNum = 2;
	end
	if( nargin < 1)
		MaxIter = 200;
	end
	
    %打开文件
    [FileName,PathName] = uigetfile('*.datset');
    if (FileName==0)
        return
    end
    file_data = load([PathName,FileName]);
	
	%解析文件
    Labels = file_data(:, 1)';
    Data = file_data(:,2:end)';
    [labelType] = unique(Labels);
    
    %创建保存目录
    dir = [PathName,'_model\'];
    if (exist(dir) ~= 0)
        rmdir(dir,'s');
    end
    mkdir(PathName, '_model');
    
    %训练每一种类，（negeLabel除外）
    for i = 1 : length(labelType)
        if (isNegLabel == 1 && labelType(i) == negLabel )
			continue;
		end
		newLabel = Labels;
		newLabel(newLabel ~= labelType(i)) = -1;
		newLabel(newLabel == labelType(i)) = 1;
		weak_learner = tree_node_w(splitNum);
        [GLearners GWeights] = GentleAdaBoost(weak_learner, Data, newLabel, MaxIter);
        fid = fopen([dir,num2str(labelType(i)), '.txt'], 'w');
        TranslateToC(GLearners, GWeights, fid, splitNum, labelType(i));
        fclose(fid);
    end

end

