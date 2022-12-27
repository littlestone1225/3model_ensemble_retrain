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
AOI_DIR=$(dirname "$SCRIPT_DIR")
echo "AOI_DIR        = "$AOI_DIR

# AOI directory name
IFS='/' read -a array <<< "$AOI_DIR"
AOI_DIR_NAME="${array[-1]}"
echo "AOI_DIR_NAME   = "$AOI_DIR_NAME


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
echo "???"

lCmdList=(
            "cd $AOI_DIR" \
            "git clone -b advan_pcb https://github.com/tkyen1110/fvcore.git" \
            "git clone -b ctr2_pcb https://github.com/tkyen1110/CenterNet2.git detectron2" \
            "git clone -b advan_pcb https://github.com/tkyen1110/darknet.git"
         )
Fun_EvalCmd "${lCmdList[*]}"
echo ""

##################################
# *****  Build detectron2  ***** #
##################################
fvcore_status=`echo $(pip list | grep fvcore) | awk -F' ' '{print $1}'`
if [ "$fvcore_status" != "fvcore" ]
then
    lCmdList=(
                "cd $AOI_DIR" \
                "pip install --user -e fvcore"
             )
    Fun_EvalCmd "${lCmdList[*]}"
else
    echo -e "${GREEN}fvcore already exists${NC}"
fi
echo ""

detectron2_status=`echo $(pip list | grep detectron2) | awk -F' ' '{print $1}'`
if [ "$detectron2_status" != "detectron2" ]
then
    lCmdList=(
                "cd $AOI_DIR" \
                "pip install --user -e detectron2"
             )
    Fun_EvalCmd "${lCmdList[*]}"
else
    echo -e "${GREEN}detectron2 already exists${NC}"
fi
echo ""

##################################
# ******   Build darknet  ****** #
##################################
darknet_status=`ls $AOI_DIR/darknet | grep libdarknet.so`
if [ -z "$darknet_status" ]
then
    lCmdList=(
                "cd $AOI_DIR/darknet" \
                "make clean" \
                "make"
             )
    Fun_EvalCmd "${lCmdList[*]}"
else
    echo -e "${GREEN}darknet already exists${NC}"
fi
echo ""

##################################
# ******   Run inference  ****** #
##################################
lCmdList=(
            "cd $AOI_DIR/config" \
            "python config.py" \
            "cd $AOI_DIR/inference" \
            "python inference.py"
         )
Fun_EvalCmd "${lCmdList[*]}"

/bin/bash
