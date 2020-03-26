#!/bin/sh

PROBLEM_NAME=at16k_subword
T2T_USR_DIR=$PWD/t2t_problems

t2t-exporter --problem=$PROBLEM_NAME \
             --model=transformer \
             --hparams_set=transformer_librispeech_tpu \
             --hparams=max_length=295650,max_input_seq_length=3650,max_target_seq_length=250 \
             --output_dir=$1 \
             --data_dir=$PWD/data \
             --t2t_usr_dir=$T2T_USR_DIR