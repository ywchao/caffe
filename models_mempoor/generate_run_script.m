prefix   = './build/tools/caffe train';

solver   = '--solver=./models_mempoor/bvlc_reference_caffenet/solver.prototxt';
snappref = '--snapshot=./models_mempoor/bvlc_reference_caffenet/models/caffenet_train_iter_';
snappost = '.solverstate';

mempoor  = '--mempoor=true';
snaptest = '--snaptest=false';

itstep     = 100;
keepitstep = 1000;
ittot      = 450000;

model_dir = 'models_mempoor/bvlc_reference_caffenet/models/';

scriptfname = 'bvlc_reference_caffenet.sh';


numIter = ittot / itstep;
snapId  = itstep : itstep : ittot;

% write to script file
fid = fopen(scriptfname,'w');

fprintf(fid,'#!/bin/bash\n');

fprintf(fid,'mkdir -p %s\n',model_dir);

for i = 1:numIter
    snap_op = [snappref(12:end) num2str(snapId(i)) snappost];
    fprintf(fid,'if [ ! -f "%s" ]; then\n',snap_op);
    if i == 1
        fprintf(fid,'  %s %s %s %s\n',prefix,solver,mempoor,snaptest);
    else
        snap_ip = [snappref num2str(snapId(i-1)) snappost];
        fprintf(fid,'  %s %s %s %s %s\n',prefix,solver,snap_ip,mempoor,snaptest);
    end
    fprintf(fid,'fi\n');
end

% clean script
fprintf(fid,'prefix="%s";\n','./models_mempoor/bvlc_reference_caffenet/models/caffenet_train_iter_');
fprintf(fid,'length=${#prefix};\n');
fprintf(fid,'keepIterStep="%d";\n',keepitstep);
fprintf(fid,'for i in `ls -tr ${prefix}*`; do\n');
fprintf(fid,'  filename=${i%%.*}  # remove extension\n');
fprintf(fid,'  iterId=${filename:$length}\n');
fprintf(fid,'  remain=$((iterId %% keepIterStep))\n');
fprintf(fid,'  if [ "$remain" -ne "0" ]; then\n');
fprintf(fid,'    rm $i;\n');
fprintf(fid,'  fi\n');
fprintf(fid,'done;\n');

fclose(fid);

cmd = ['chmod 755 ' scriptfname];
system(cmd);
