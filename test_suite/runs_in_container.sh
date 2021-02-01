#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate rapids

cd /hpf/largeprojects/adam/projects/raccoon/tests

time python3 run_tests.py
