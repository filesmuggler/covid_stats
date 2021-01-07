#!/bin/bash
: '
Downloads data from given github folder
Autor: Krzysztof Stezala <krzysztof.stezala at student.put.poznan.pl>
Version: 0.1
License: MIT
'
REPO_LINK=$1

DIR="data"
TRUNK="trunk"
if [ -d "$DIR" ]; then
  # Take action if $DIR exists. #
  echo "Directory ${DIR} exists."
  echo "Updating ${DIR} directory.."
  svn export --force "${REPO_LINK/"tree/master"/$TRUNK}" ${DIR}
else
  echo "Creating ${DIR} directory..."
  mkdir data
  echo "Copying data to ${DIR} directory.."
  svn export --force "${REPO_LINK/"tree/master"/$TRUNK}" ${DIR}
fi
