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


# # # # # # # # # # # # # # # # # # # # # # # # #
#     Download dataset by Datasource SDK        #
# # # # # # # # # # # # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR" \
            "python data_preprocess/s3_manip.py"
         )
# Fun_EvalCmd "${lCmdList[*]}"


# # # # # # # # # # # # # # # # # # # # #
#     Download model by Model SDK       #
# # # # # # # # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/data_preprocess" \
            "python model_repo_manip.py"
         )
# Fun_EvalCmd "${lCmdList[*]}"


# # # # # # # # # # # # # # # # #
#     Global Configuration      #
# # # # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/config" \
            "python config.py"
         )
Fun_EvalCmd "${lCmdList[*]}"


# # # # # # # # # # # # # # #
#     Data Preparation      #
# # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/data_preprocess" \
            "python prepare_retrain_data.py" \
         )
#Fun_EvalCmd "${lCmdList[*]}"

lCmdList=(
            "cd $AOI_PCB_DIR/data_preprocess" \
            "python prepare_train_data.py" \
            "python prepare_test_data.py" \
            "python prepare_val_data.py" 
         )
#Fun_EvalCmd "${lCmdList[*]}"
