#!/bin/sh

if [ -z "$TRAVIS_TAG" ]; then
    DEPLOY_DIR=dev;
else
    DEPLOY_DIR="version-$TRAVIS_TAG";
fi

make -C apidocs html

doctr deploy --no-require-master --build-tags --command "$VIRTUAL_ENV/bin/python build_index.py" --built-docs apidocs/_build/html $DEPLOY_DIR
