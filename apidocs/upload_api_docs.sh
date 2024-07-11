#!/bin/sh

set -ex

# Clone gh-pages
git clone --branch gh-pages https://github.com/bmcage/odes gh-pages
# Run rsync
branch_name="$GITHUB_REF_NAME"
rsync -av --delete "$API_DOCS_OUT_DIR" "./gh-pages/$branch_name/"
# Run docs-versions-menu
cd gh-pages
docs-versions-menu
# Commit and push
git config user.name github-actions
git config user.email github-actions@github.com
git add -A --verbose
git status
git commit --verbose -m "Auto-update from Github Actions Workflow" -m "Deployed from commit ${GITHUB_SHA} (${GITHUB_REF})"
git push --verbose --force "https://$GITHUB_ACTOR:$GITHUB_TOKEN@github.com/bmcage" gh-pages