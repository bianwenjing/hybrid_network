ROOT=..

#export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export MESHFUSION_PATH=/home/wenjing/occupancy_networks-master/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

#INPUT_PATH=$ROOT/data/external/ShapeNetCore.v1
#INPUT_PATH=/home/wenjing/storage/raw/ShapeNetCore.v1
#INPUT_PATH=/home/wenjing/Downloads/ShapeNetCore.v1
INPUT_PATH=/media/wenjing/Data21/ShapeNetCore.v1
#CHOY2016_PATH=$ROOT/data/external/Choy2016
CHOY2016_PATH=/home/wenjing/storage/
#BUILD_PATH=/home/wenjing/storage/raw/ShapeNet.build2
BUILD_PATH=/media/wenjing/Data21/ShapeNet.build
#BUILD_PATH=/media/wenjing/data3/ShapeNet.build
OUTPUT_PATH=/home/wenjing/Downloads/ShapeNet/
#OUTPUT_PATH=/home/wenjing/storage/data/1d_ShapeNet/2d_ShapeNet

NPROC=12
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

declare -a CLASSES=(
#03001627
#02958343
#04256520
#02691156
#03636649 #lamp
#04401088
#04530566
#03691459
#02933112
#04379243
#03211117
02828884
#04090263
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}
