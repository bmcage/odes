#!/bin/sh
set -ex

wget http://computation.llnl.gov/projects/sundials-suite-nonlinear-differential-algebraic-equation-solvers/download/sundials-2.6.2.tar.gz

tar -xzvf sundials-2.6.2.tar.gz

mkdir sundials_build

cd sundials_build &&
    cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=ON ../sundials-2.6.2 &&
    make && make install
