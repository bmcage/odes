#!/bin/sh

export SUNDIALS_DIR=$HOME/sundials-"${SUNDIALS_VERSION:-'2.6.2'}"
SUNDIALS_LIBDIR=$SUNDIALS_DIR/lib
SUNDIALS_INCLUDEDIR=$SUNDIALS_DIR/include

if [ ! -d "$SUNDIALS_LIBDIR" ]; then
    mkdir -p $SUNDIALS_DIR
    echo "Installing sundials"
    ./ci_support/install_sundials.sh
else
    echo "Using cached sundials"
fi

export LD_LIBRARY_PATH=$SUNDIALS_LIBDIR:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SUNDIALS_LIBDIR:$LIBRARY_PATH
export CPATH=$SUNDIALS_INCLUDEDIR:$CPATH
