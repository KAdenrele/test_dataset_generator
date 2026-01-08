#!bin/bash

docker build -t test-data-app -f Dockerfile .
#docker run --rm -it -v /Users/akomolafe/datasets:/data test-data-app