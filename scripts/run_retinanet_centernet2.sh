#!/bin/bash
# Absolute path to this script file
SCRIPT_FILE=$(readlink -f "$0")

# Absolute directory this script is in
SCRIPT_DIR=$(dirname "$SCRIPT_FILE")

# Absolute path to the AOI_PCB_DIR
AOI_PCB_DIR=$(dirname "$SCRIPT_DIR")

# Get config.yaml
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
printf "${GREEN}cd $AOI_PCB_DIR/config${NC}\n"
cd $AOI_PCB_DIR/config

printf "${GREEN}python config.py --set_production${NC}\n"
config_result=`python config.py --set_production`
Fun_ConvertConfigResult "$config_result" 1

if [ "$PRODUCTION" != "retrain_aifs" ]
then
    # Color
    NC='\033[0m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
fi


# # # # # # # # # # # # # # # #
#     RetinaNet Training      #
# # # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/detectron2" \
            "CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file ./configs/COCO-Detection/retinanet_R_101_FPN_3x_PCB.yaml --num-gpus 1"
         )
Fun_EvalCmd "${lCmdList[*]}"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     CenterNet2 Training, Validation and Evaluation        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/detectron2" \
            "CUDA_VISIBLE_DEVICES=0 python projects/CenterNet2/retrain_net.py --num-gpus 1 --aifs"
         )
Fun_EvalCmd "${lCmdList[*]}"

# # # # # # # # # # # # # # # # # # # # # # # # #
#     RetinaNet Validation and Evaluation       #
# # # # # # # # # # # # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/validation" \
            "python val_and_eval.py"
         )
Fun_EvalCmd "${lCmdList[*]}"


