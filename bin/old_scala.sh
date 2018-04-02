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
# sysnet path
SYNSET=$4
# number of runs
NUM_RUN=$5

CLASSPATH=$CLASSPATH:/Users/roshanin/mxnet/scala-package/assembly/osx-x86_64-cpu/target/*:$CLASSPATH:/Users/roshanin/workspace/benchmark_scala/target/*:$CLASSPATH:/Users/roshanin/workspace/benchmark_scala/target/classes/lib/*
java -Xmx8G -cp $CLASSPATH \
  me.roshani.mxnet.scala.benchmark.OldScalaBench \
  	--model-path-prefix $MODEL_PATH_PREFIX \
	--input-image $INPUT_IMG \
	--input-dir $INPUT_DIR \
	--synset $SYNSET \
	--num-run $NUM_RUN
