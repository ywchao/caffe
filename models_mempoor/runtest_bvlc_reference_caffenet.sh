# $1: gpu id

runtestintv=1000;

./build/tools/caffe train --solver=./models_mempoor/bvlc_reference_caffenet/solver.prototxt --gpu=$1 --mempoor=true --runtest=true --runtestintv=${runtestintv} 2>&1 | tee ./models_mempoor/bvlc_reference_caffenet/runtest.log;


