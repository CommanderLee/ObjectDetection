function TrainAndSaveEx( MaxIter, splitNum, negLabel )  
%TrainAndSaveEx( MaxIter, splitNum, negLabel )
%   MaxIter: ����������,Ĭ��200
%   splitNum: ���ֲ����,Ĭ��Ϊ2
%   negLabel: ��������ǩ,Ĭ�ϲ�ָ��    

    %Ĭ�ϲ���
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
	
    %���ļ�
    [FileName,PathName] = uigetfile('*.datset');
    if (FileName==0)
        return
    end
    file_data = load([PathName,FileName]);
	
	%�����ļ�
    Labels = file_data(:, 1)';
    Data = file_data(:,2:end)';
    [labelType] = unique(Labels);
    
    %��������Ŀ¼
    dir = [PathName,'_model\'];
    if (exist(dir) ~= 0)
        rmdir(dir,'s');
    end
    mkdir(PathName, '_model');
    
    %ѵ��ÿһ���࣬��negeLabel���⣩
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

