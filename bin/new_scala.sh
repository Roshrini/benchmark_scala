#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)
DATA_DIR=$PROJ_DIR/data

# model dir
MODEL_PATH_PREFIX=$1
# input image
INPUT_IMG=$2
# which input image dir
INPUT_DIR=$3
# number of runs
NUM_RUN=$4

CLASSPATH=$CLASSPATH:/Users/roshanin/mxnet/scala-package/assembly/osx-x86_64-cpu/target/*:$CLASSPATH:/Users/roshanin/workspace/benchmark_scala/target/*:$CLASSPATH:/Users/roshanin/workspace/benchmark_scala/target/classes/lib/*:$CLASSPATH:/Users/roshanin/mxnet/scala-package/infer/target/*
java -Xmx8G -cp $CLASSPATH \
  me.roshani.mxnet.scala.benchmark.InferBenchmark \
  	--model-path-prefix $MODEL_PATH_PREFIX \
	--input-image $INPUT_IMG \
	--input-dir $INPUT_DIR \
	--num-run $NUM_RUN
