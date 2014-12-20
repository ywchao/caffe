prefix   = './build/tool/caffe train';

solver   = '--solver=./models_mempoor/bvlc_reference_caffenet/solver.prototxt';
snappref = '--snapshot=./models/mempoor/bvlc_reference_caffenet/models/caffenet_train_iter_';
snappost = '.solverstate';

mempoor  = '--mempoor=true';
snaptest = '--snaptest=false';

itstep = 100;
ittot  = 450000;

scriptfname = 'train_bvlc_reference_caffenet.sh';


numIter = ittot / itstep;
snapId  = 0 : itstep : ittot - itstep;

% write to script file
fid = fopen(scriptfname,'w');

for i = 1:numIter
    if i == 1
        fprintf(fid,'%s %s %s %s\n',prefix,solver,mempoor,snaptest);
    else
        snap = [snappref num2str(snapId(i)) snappost];
        fprintf(fid,'%s %s %s %s %s\n',prefix,solver,snap,mempoor,snaptest);
    end
end

fclose(fid);

