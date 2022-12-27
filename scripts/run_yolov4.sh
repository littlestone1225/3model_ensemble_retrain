#!/bin/bash
# Absolute path to this script file
SCRIPT_FILE=$(readlink -f "$0")

# Absolute directory this script is in
SCRIPT_DIR=$(dirname "$SCRIPT_FILE")

# Absolute path to the AOI_PCB directory
AOI_PCB_DIR=$(dirname "$SCRIPT_DIR")

# Get config from config.py
function Fun_ConvertConfigResult()
{
    config_result=$1
    print_result=$2
    config_result=`echo $config_result | sed 's/{//g' | sed 's/}//g' | sed 's/'\''//g'`
    IFS=",: " read -a config <<< $config_result

    for (( i=0; i<${#config[@]}; i+=2 ))
    do
        key=`echo ${config[$i]^^} | sed -e 's/\r//g'`
        value=`echo ${config[$i+1]} | sed -e 's/\r//g'`
        eval "$key=$value"
        if [ "$print_result" = "1" ]
        then
            printf "${YELLOW}%-20s = $value${NC}\n" $key
        fi
    done
}

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}\n${cmd}${NC}\n"
        eval $cmd
        exit_code=$?

        if [[ $exit_code = 0 ]]; then
            printf "${GREEN}[Success] ${cmd} ${NC}\n"
        else
            printf "${RED}[Failure: $exit_code] ${cmd} ${NC}\n"
            exit 1
        fi
    done
}


# # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Set production in global configuration        #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
printf "${GREEN}python $AOI_PCB_DIR/config/config.py --set_production${NC}\n"
config_result=`python $AOI_PCB_DIR/config/config.py --set_production`
Fun_ConvertConfigResult "$config_result" 1

if [ "$PRODUCTION" != "retrain_aifs" ]
then
    # Color
    NC='\033[0m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
fi


# 1. generate yolov4 config yaml
printf "${GREEN}cd $AOI_PCB_DIR/YOLOv4${NC}\n"
cd $AOI_PCB_DIR/YOLOv4

printf "${GREEN}python yolo_config_yaml.py${NC}\n"
config_result=`python yolo_config_yaml.py`
# config_result=`echo $config_result | awk -F'{' '{print $2}' | awk -F'}' '{print $1}'`
Fun_ConvertConfigResult "$config_result" 1


# 2. retrain preprocess
lCmdList=(
            "python3 yolov4_pre_process.py"
         )
Fun_EvalCmd "${lCmdList[*]}"


# 3. yolov4 training
lCmdList=(
            "$YOLO_DARKNET_PATH/darknet detector train $YOLO_CONFIG_PATH/pcb.data $YOLO_CONFIG_PATH/yolov4-pcb.cfg $YOLOV4_OLD_MODEL_FILE_PATH -dont_show -clear 1 -gpus 1"
         )
Fun_EvalCmd "${lCmdList[*]}"



# 4. valid & test
lCmdList=(
            "python3 valid_and_test.py"
         )
Fun_EvalCmd "${lCmdList[*]}"
