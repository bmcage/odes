#!/bin/sh

set -e

if [ "$GITHUB_REPOSITORY" = "bmcage/odes" ] && [ "$DOCTR_DEPLOY" ]; then
    deploy="true"
else
    deploy="false"
fi


if [ "tag" = "$GITHUB_REF_TYPE" ]; then
    DEPLOY_DIR=dev;
else
    DEPLOY_DIR="version-$GITHUB_REF_NAME";
fi

make -C apidocs html

if [ $deploy = "true" ]; then
    doctr deploy --no-require-master --build-tags --command "$VIRTUAL_ENV/bin/python build_index.py" --built-docs apidocs/_build/html "$DEPLOY_DIR"
else
    echo 'Not deploying docs, set DOCTR_DEPLOY to do deployment of docs'
fi
