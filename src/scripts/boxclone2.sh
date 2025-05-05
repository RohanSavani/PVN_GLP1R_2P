#!/bin/bash

run_copy() {
    local CMD="$1"
    echo "Starting: $CMD"
    eval "$CMD"
    echo "Finished: $CMD"
}

COMMANDS=(
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/glp23_fed_30suc_L155P800_575um-20250203-112' './glp23_fed_30suc_L155P800_575um-20250203-112' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/glp23_wd_30suc_L155P800_565um-107' './glp23_wd_30suc_L155P800_565um-107' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/glp23_fasted_50suc+sucra_L155P800_575um-20250206-114' './glp23_fasted_50suc+sucra_L155P800_575um-20250206-114' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/glp23_fasted_50suc+wat_L155P800_575um-20250207-134' './glp23_fasted_50suc+wat_L155P800_575um-20250207-134' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP23/glp23_water-fasted_60suc+qui_L155P800_575um-20250208-151' './glp23_water-fasted_60suc+qui_L155P800_575um-20250208-151' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/glp17_fasted30suc_L150P800_565um-032' './glp17_fasted30suc_L150P800_565um-032' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/glp17_wd_30trials_565um_L150P800-074' './glp17_wd_30trials_565um_L150P800-074' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP17/glp17_fed30suc_L150P800_565um_0613-020' './glp17_fed30suc_L150P800_565um_0613-020' -vP"
    # "rclone copy 'box:Dump/6/glp17_25suc25sucralose_fasted_L150P800_565um-061' './glp17_25suc25sucralose_fasted_L150P800_565um-061' -vP"
    # "rclone copy 'box:Dump/6/glp17_30suc20water_fasted_L150P800_565um-003' './glp17_30suc20water_fasted_L150P800_565um-003' -vP"
    # "rclone copy 'box:glp17_wd_45suc15quinine(0.5mM)_L150P800_565um-017' './glp17_wd_45suc15quinine(0.5mM)_L150P800_565um-017' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/No Cue Data/GLP06_FedNoCues_L180P815_600um-03062024-1321-073' './GLP06_FedNoCues_L180P815_600um-03062024-1321-073' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/No Cue Data/GLP6_FastedNoCues_L180_P815_600um-_-03092024-1421-084' './GLP6_FastedNoCues_L180_P815_600um-_-03092024-1421-084' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/No Cue Data/GLP6_waterdNoCues_L170_P800_590um-03042024-1137-058' './GLP6_waterdNoCues_L170_P800_590um-03042024-1137-058' -vP"
    "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Other Tastants/glp6_wd_L150P800600um_suc0.5mmquinine_032324' './glp6_wd_L150P800600um_suc0.5mmquinine_032324' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Other Tastants/glp6_25sucralose25sucrose_fasted_L170P800_600um_031324-003' './glp6_25sucralose25sucrose_fasted_L170P800_600um_031324-003' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m_#6/Other Tastants/glp6_fasted_sucwater_nocues_L170_p800_600um031124-006' './glp6_fasted_sucwater_nocues_L170_p800_600um031124-006' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP10/glp10_WD_45suc15quinine(0.5mM)_L170P800_550um_04132024-1218-024' './glp10_WD_45suc15quinine(0.5mM)_L170P800_550um_04132024-1218-024' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP10/glp10_fasted_25suc25sucralose_L170P800_550um_04124-010' './glp10_fasted_25suc25sucralose_L170P800_550um_04124-010' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP10/glp10_fasted_30suc20water_L170P800_550um_041024-006' './glp10_fasted_30suc20water_L170P800_550um_041024-006' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP10/glp10_fed_30suc_L170P800_560um_040924-025' './glp10_fed_30suc_L170P800_560um_040924-025' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP10/glp10_fasted_30suc_L170P800_560um_040724-005' './glp10_fasted_30suc_L170P800_560um_040724-005' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GLP10/GLP10_240404_WD_suc_Dep625um_L170_T800_-007' './GLP10_240404_WD_suc_Dep625um_L170_T800_-007' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/No Cue Data/glp4_fasted_nocues_L180_p815_650um031124-003' './glp4_fasted_nocues_L180_p815_650um031124-003' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/No Cue Data/GLP4_FedNoCues_L180_P815_650um-_03052024-1355-066' './GLP4_FedNoCues_L180_P815_650um-_03052024-1355-066' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/No Cue Data/GLP4_waterdNoCues_L180_P815_650um-03042024-1137-056' './GLP4_waterdNoCues_L180_P815_650um-03042024-1137-056' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/Other Tastants/glp4_25sucrose_25sucralose_fasted_L180P815_650um--03202024-005' './glp4_25sucrose_25sucralose_fasted_L180P815_650um--03202024-005' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/Other Tastants/glp4_wd_L180P815650um_suc0.5mmquinine_032324-005' './glp4_wd_L180P815650um_suc0.5mmquinine_032324-005' -vP"
    # "rclone copy 'box:(Restricted) Pang Lab Data 2/Pang lab data/PVN in vivo 2P/Data from Le/GRC_PVN_GCaMP8m#4/Other Tastants/glp4_fasted_sucwater_L180P815_650um_031824-004' './glp4_fasted_sucwater_L180P815_650um_031824-004' -vP"
)

JOB_COUNT=0
MAX_JOBS=2  # Limit number of concurrent jobs

# Loop through the commands and start them in parallel
# for CMD in "${COMMANDS[@]}"; do
#     # Extract directory name from the command using a different method to avoid issues with path extraction
#     DIR=$(echo "$CMD" | sed -n "s/.*'\(.*\)'.*/\1/p" | awk -F' ' '{print $1}')
#     
#     # Create the destination folder if it doesn't exist
#     echo "Creating directory: $DIR"
#     mkdir -p "$DIR"
#     
#     # Run the rclone copy command in the background
#     run_copy "$CMD" &
#     
#     ((JOB_COUNT++))
#     
#     # Limit the number of parallel jobs
#     if (( JOB_COUNT >= MAX_JOBS )); then
#         wait -n  # Wait for any job to finish
#         ((JOB_COUNT--))
#     fi
# done
# 
# # Wait for remaining jobs to finish
# wait
# 
# echo "All downloads completed!"


# Loop through the commands and run them sequentially
for CMD in "${COMMANDS[@]}"; do
    echo "Starting: $CMD"
    
    # Extract directory name from the command
    DIR=$(echo "$CMD" | grep -o "'./[^']*'" | tr -d "'")
    
    # Create the destination folder if it doesn't exist
    mkdir -p "$DIR"
    
    # Run the rclone copy command
    eval "$CMD"
    
    echo "Finished: $CMD"
done

echo "All downloads completed!"


