#!/bin/sh
set -ex

SUNDIALS=sundials-"${SUNDIALS_VERSION:-5.1.0}"
SUNDIALS_FILE=$SUNDIALS.tar.gz
SUNDIALS_URL=https://github.com/LLNL/sundials/releases/download/v${SUNDIALS_VERSION}/${SUNDIALS_FILE}
PRECISION="${SUNDIALS_PRECISION:-double}"
INDEX_SIZE="${SUNDIALS_INDEX_SIZE:-64}"

wget "$SUNDIALS_URL"

tar -xzvf "$SUNDIALS_FILE"

mkdir sundials_build

if [ "$PRECISION" = "extended" ]; then
    cd sundials_build &&
        cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=OFF -DSUNDIALS_INDEX_SIZE="$INDEX_SIZE" -DSUNDIALS_PRECISION="$PRECISION" -DEXAMPLES_INSTALL:BOOL=OFF ../$SUNDIALS &&
        make && make install
else 
    cd sundials_build &&
        cmake -DCMAKE_INSTALL_PREFIX=$SUNDIALS_DIR -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_SIZE="$INDEX_SIZE" -DSUNDIALS_PRECISION="$PRECISION" -DEXAMPLES_INSTALL:BOOL=OFF ../$SUNDIALS &&
        make && make install
fi
