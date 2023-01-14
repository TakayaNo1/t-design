#!/bin/sh

#create directory
mkdir ./data_gen/result
mkdir ./fp/result
mkdir ./ml/datasets

#download and build qulacs
git clone https://github.com/qulacs/qulacs.git
cd qulacs
#./script/build_gcc.sh
