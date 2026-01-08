#!/bin/bash
#source = https://github.com/cvdfoundation/kinetics-dataset/blob/main/k400_extractor.sh
# Download directories vars
root_dl="/data/raw/k400"
root_dl_targz="/data/raw/k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl


# Extract validation
curr_dl=$root_dl_targz/val
curr_extract=$root_dl/val
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done


# Extract replacement
curr_dl=$root_dl_targz/replacement
curr_extract=$root_dl/replacement
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tgz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extraction complete
echo -e "\nExtractions complete!"