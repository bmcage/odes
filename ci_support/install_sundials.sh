#!/bin/sh
set -ex

wget http://computation.llnl.gov/projects/sundials/download/sundials-2.7.0.tar.gz

tar -xzvf sundials-2.7.0.tar.gz

mkdir sundials_build

cd sundials_build &&
    cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=ON ../sundials-2.7.0 &&
    make && make install
