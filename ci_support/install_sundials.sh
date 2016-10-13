#!/bin/sh
set -ex

SUNDIALS=sundials-"${SUNDIALS_VERSION:-'2.7.0'}"
SUNDIALS_FILE=$SUNDIALS.tar.gz
SUNDIALS_URL=http://computation.llnl.gov/projects/sundials-suite-nonlinear-differential-algebraic-equation-solvers/download/$SUNDIALS_FILE

wget "$SUNDIALS_URL"

tar -xzvf "$SUNDIALS_FILE"

mkdir sundials_build

cd sundials_build &&
    cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=ON ../$SUNDIALS &&
    make && make install
