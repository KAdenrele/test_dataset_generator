#!bin/bash
docker run --rm -it \
  -v /Users/akomolafe/my_datasets:/data \
  --entrypoint /bin/bash \
  test-data-app


# ./bash_scripts/k400_download.sh
# ./bash_scripts/k400_extraction.sh
# python main.py