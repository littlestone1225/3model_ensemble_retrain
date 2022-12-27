#!/bin/bash

# Color
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

# Absolute path to this script.
# e.g. /home/ubuntu/AOI_PCB_Inference/dockerfile/dockerfile_inference.sh
SCRIPT_PATH=$(readlink -f "$0")

# Absolute path this script is in.
# e.g. /home/ubuntu/AOI_PCB_Inference/dockerfile
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Absolute path to the AOI path
# e.g. /home/ubuntu/AOI_PCB_Inference
HOST_AOI_DIR=$(dirname "$SCRIPT_DIR")
echo "HOST_AOI_DIR  = "$HOST_AOI_DIR

# AOI directory name
IFS='/' read -a array <<< "$HOST_AOI_DIR"
AOI_DIR_NAME="${array[-1]}"
echo "AOI_DIR_NAME   = "$AOI_DIR_NAME


VERSION=$2
if [ "$2" == "" ]
then
    VERSION="v1.0"
else
    VERSION=$2
fi
echo "VERSION        = "$VERSION

IMAGE_NAME="pcb_inference_3models:$VERSION"
CONTAINER_NAME="pcb_inference_3models_$VERSION"
echo "IMAGE_NAME     = "$IMAGE_NAME
echo "CONTAINER_NAME = "$CONTAINER_NAME

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}${cmd}${NC}\n"
        eval $cmd
    done
}


if [ "$1" == "build" ]
then
    export GID=$(id -g)

    lCmdList=(
                "docker build \
                    --build-arg USER=$USER \
                    --build-arg UID=$UID \
                    --build-arg GID=$GID \
                    --build-arg AOI_DIR_NAME=$AOI_DIR_NAME \
                    -f pcb_retrain.dockerfile \
                    -t $IMAGE_NAME ."
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "run" ]
then
    HOST_API_PORT="80"

    lCmdList=(
                "docker run --gpus all -itd \
                    --privileged \
                    --restart unless-stopped \
                    --ipc=host \
                    --name $CONTAINER_NAME \
                    -v $HOST_AOI_DIR:/home/$USER/$AOI_DIR_NAME \
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /etc/localtime:/etc/localtime:ro \
                    --mount type=bind,source=$SCRIPT_DIR/.bashrc,target=/home/$USER/.bashrc \
                    $IMAGE_NAME $HOME/$AOI_DIR_NAME/dockerfile/env_setup.sh" \
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "exec" ]
then
    lCmdList=(
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "start" ]
then
    lCmdList=(
                "docker start -ia $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "attach" ]
then
    lCmdList=(
                "docker attach $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "stop" ]
then
    lCmdList=(
                "docker stop $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rm" ]
then
    lCmdList=(
                "docker rm $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rmi" ]
then
    lCmdList=(
                "docker rmi $IMAGE_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"
    
elif [ "$1" = "hard_clean" ]
then
    lCmdList=(
                "sudo rm -r ../detectron2/" \
                "sudo rm -r ../fvcore/" \
                "sudo rm -r ../darknet/" \
                "sudo rm ../config/config.yaml.lock"
             )
    Fun_EvalCmd "${lCmdList[*]}"
elif [ "$1" = "data_clean" ]
then
    lCmdList=(
                "sudo rm -r ../pcb_data/" \
                "sudo rm -r ../pcb-dataset/train_data/images_sliding_crop_mask_margin/" \
                "sudo rm -r ../pcb-dataset/train_data/labels_sliding_crop_mask_margin/" \
                "sudo rm -r ../pcb-dataset/train_data/shrink_images_wo_border/" \
                "sudo rm -r ../pcb-dataset/train_data/images_random_crop_w_aug/" \
                "sudo rm -r ../pcb-dataset/train_data/labels_random_crop_w_aug/" \
                "sudo rm -r ../pcb-dataset/train_data/images_wo_border/" \
                "sudo rm -r ../pcb-dataset/train_data/labels_wo_border/" \
                "sudo rm -r ../pcb-dataset/test_data/images_random_crop/" \
                "sudo rm -r ../pcb-dataset/test_data/labels_random_crop/" \
                "sudo rm -r ../pcb-dataset/test_data/images_wo_border/" \
                "sudo rm -r ../pcb-dataset/test_data/labels_wo_border/" \
                "sudo rm -r ../pcb-dataset/val_data/images_random_crop/" \
                "sudo rm -r ../pcb-dataset/val_data/labels_random_crop/" \
                "sudo rm -r ../pcb-dataset/val_data/images_wo_border/" \
                "sudo rm -r ../pcb-dataset/val_data/labels_wo_border/" \
                "sudo rm -r ../pcb-dataset/retrain_data/train/" \
                "sudo rm -r ../pcb-dataset/retrain_data/val/" \
                "sudo rm -r ../pcb-dataset/retrain_data/original_preprocess/" \
                "sudo rm -r ../config/__pycache__/"\
                "sudo rm -r ../data_preprocess/__pycache__/"\
                "sudo rm -r ../ensemble/__pycache__/"\
                "sudo rm -r ../inference/__pycache__/"\
                "sudo rm -r ../utils/__pycache__/"\
                "sudo rm -r ../validation/__pycache__/"\
                "sudo rm -r ../YOLOv4/__pycache__/"
             )
    Fun_EvalCmd "${lCmdList[*]}"
fi