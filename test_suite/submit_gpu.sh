#!/bin/bash
#PBS -S /bin/bash
#PBS -V
#PBS -l vmem=170g,mem=170g
#PBS -l walltime=240:00:00
#PBS -l nodes=1:ppn=18:gpus=1
#PBS -N rc_tests_gpu
#PBS -q gpu

cd /hpf/largeprojects/adam/projects/raccoon/tests

module load rapids

singularity exec -B /hpf --nv /hpf/tools/centos7/rapids/0.16/rapidsai_cuda10.1-runtime-centos7-py3.8.sif ./runs_in_container.sh
