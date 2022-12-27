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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       Trap SIGINT and SIGTERM to stop child processes     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
killpids() {
    kpids=`jobs -p`
    for pid in ${kpids[*]}; do
        printf "${GREEN}pkill -P $pid${NC}\n"
        pkill -P $pid
    done
    exit 1
}
trap killpids SIGINT
trap killpids SIGTERM


# # # # # # # # # # # # # # #
#       run_prepare.sh      #
# # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/scripts" \
            "bash run_prepare.sh"
         )
Fun_EvalCmd "${lCmdList[*]}"


# # # # # # # # # # # # # # # # # # # # #
#       run_retinanet_centernet2.sh     #
#       run_yolov4.sh                   #
# # # # # # # # # # # # # # # # # # # # #
set -o monitor # make "Interrupt" key work. ex: ctrl-C

if [ "$PRODUCTION" == "retrain_aifs" ]
then
    lCmdList=(
                "cd $AOI_PCB_DIR/scripts" \
                "bash run_retinanet_centernet2.sh &" \
                "bash run_yolov4.sh > /dev/null 2>&1 &"
             )
else
    lCmdList=(
                "cd $AOI_PCB_DIR/scripts" \
                "bash run_retinanet_centernet2.sh | tee run_retinanet_centernet2.log &" \
                "bash run_yolov4.sh > run_yolov4.log 2>&1 &"
             )
fi
Fun_EvalCmd "${lCmdList[*]}"


# https://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0?page=1&tab=votes#tab-top
pids=`jobs -p`
for pid in ${pids[*]}; do
    printf "${GREEN}PID = $pid is running${NC}\n"
done

checkpids() {
    for pid in $pids; do
        if kill -0 $pid 2>/dev/null;
        then
            printf "${GREEN}PID = $pid is still alive${NC}\n"
        elif wait $pid; then
            printf "${GREEN}PID = $pid exited with zero exit status${NC}\n"
        else
            printf "${RED}PID = $pid exited with non-zero exit status${NC}\n"

            for pid_t in $pids; do
                if [ $pid_t = $pid ]; then
                    continue
                else
                    pid_t=$(ps -ef | grep $pid_t | grep -v grep | awk '{print $2}' | grep $pid_t)
                    if [ ! -z $pid_t ]; then
                        printf "${GREEN}pkill -P $pid_t${NC}\n"
                        pkill -P $pid_t
                    fi
                fi
            done
            break
        fi
    done
}

trap checkpids SIGCHLD
# wait $pids

for pid in $pids; do
    wait $pid
    exit_code=$?
    if [ $exit_code != 0 ]
    then
        exit 1
    fi
done
trap '' SIGCHLD

# # # # # # # # # # # # # # #
#       run_ensemble.sh     #
# # # # # # # # # # # # # # #
lCmdList=(
            "cd $AOI_PCB_DIR/scripts" \
            "bash run_ensemble.sh"
         )
Fun_EvalCmd "${lCmdList[*]}"
