#!/bin/sh

set -e

if [ "$TRAVIS_REPO_SLUG" = "bmcage/odes" -o "$DOCTR_DEPLOY" ]; then
    deploy="true"
else
    deploy="false"
fi


if [ -z "$TRAVIS_TAG" ]; then
    DEPLOY_DIR=dev;
else
    DEPLOY_DIR="version-$TRAVIS_TAG";
fi

make -C apidocs html

if [ $deploy = "true" ]; then
    doctr deploy --no-require-master --build-tags --command "$VIRTUAL_ENV/bin/python build_index.py" --built-docs apidocs/_build/html $DEPLOY_DIR
else
    echo 'Not deploying docs, set DOCTR_DEPLOY to do deployment of docs'
fi
