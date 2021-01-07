#!/bin/bash
: '
Downloads data from CSSE repo
Autor: Krzysztof Stezala <krzysztof.stezala at student.put.poznan.pl>
Version: 0.1
License: MIT
'
DIR="data"
if [ -d "$DIR" ]; then
  # Take action if $DIR exists. #
  echo "Directory ${DIR} exists."
  echo "Updating ${DIR} directory.."
  svn export --force https://github.com/CSSEGISandData/COVID-19/trunk/csse_covid_19_data/csse_covid_19_time_series ${DIR}
else
  echo "Creating ${DIR} directory..."
  mkdir data
  echo "Copying data to ${DIR} directory.."
  svn export --force https://github.com/CSSEGISandData/COVID-19/trunk/csse_covid_19_data/csse_covid_19_time_series ${DIR}
fi
