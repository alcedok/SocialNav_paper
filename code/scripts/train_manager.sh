#!/bin/bash
# export PYTHONPATH=$(pwd)/code

LOG_DIR="logs"
NUM_INSTANCES=5  # number of jobs to run
PYTHON_SCRIPT="scripts/policy_learning_experiments.py"  # script to run

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Function to start training jobs
start_jobs() {
    if [[ -z "$1" || -z "$2" ]]; then
        echo "Usage: $0 start <environment_type> <influence_mode>"
        exit 1
    fi

    ENVIRONMENT_TYPE="$1"
    INFLUENCE_MODE="$2"
    PID_FILE="train_pids_${ENVIRONMENT_TYPE}_${INFLUENCE_MODE}.txt"

    > "$PID_FILE"  # Clear previous PID file for this variable combination

    for i in $(seq 1 $NUM_INSTANCES); do
        TIMESTAMP=$(date +'%d-%m-%y_%H:%M:%S')
        INSTANCE_ID=$(date +'%s%N' | tail -c 7)
        LOG_FILE="$LOG_DIR/${ENVIRONMENT_TYPE}_${INFLUENCE_MODE}_exp-${i}-$(date +'%s').log"

        nohup python "$PYTHON_SCRIPT" \
            --environment_type "$ENVIRONMENT_TYPE" \
            --influence_mode "$INFLUENCE_MODE" \
            --timestamp "$TIMESTAMP" \
            --instance_id "$INSTANCE_ID" \
            --logfile "$LOG_FILE" > "$LOG_FILE" 2>&1 &
        PID=$!
        
        echo "$PID $ENVIRONMENT_TYPE $INFLUENCE_MODE \"$TIMESTAMP\" \"$INSTANCE_ID\" \"$LOG_FILE\"" >> "$PID_FILE"
        echo "Started instance $i (PID: $PID) for experiment '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE' at $TIMESTAMP"
    done

    echo "All jobs for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE' started."
    echo "Use '$0 list <environment_type> <influence_mode>' to see running jobs."
}

# Function to list running jobs
list_jobs() {
    if [[ -z "$1" || -z "$2" ]]; then
        echo "Usage: $0 list <environment_type> <influence_mode>"
        exit 1
    fi

    ENVIRONMENT_TYPE="$1"
    INFLUENCE_MODE="$2"
    PID_FILE="train_pids_${ENVIRONMENT_TYPE}_${INFLUENCE_MODE}.txt"

    if [[ ! -f "$PID_FILE" || ! -s "$PID_FILE" ]]; then
        echo "No active training jobs found for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE'."
        exit 0
    fi

    printf "%-8s | %-15s | %-20s | %-20s | %-15s | %-50s\n" "PID" "Environment" "Influence Mode" "Start Time" "Instance ID" "Log File"
    printf "%-8s | %-15s | %-20s | %-20s | %-15s | %-50s\n" "--------" "---------------" "--------------------" "--------------------" "---------------" "--------------------------------------------------"
    while read -r PID ENVIRONMENT_TYPE INFLUENCE_MODE START_TIME INSTANCE_ID LOG_FILE; do
        if ps -p "$PID" > /dev/null 2>&1; then
            printf "%-8s | %-15s | %-20s | %-15s | %-15s | %-50s\n" "$PID" "$ENVIRONMENT_TYPE" "$INFLUENCE_MODE" "$START_TIME" "$INSTANCE_ID" "$LOG_FILE"
        fi
    done < "$PID_FILE"
}

# Function to list all running jobs across all experiments
list_all_jobs() {
    echo "Listing all running experiments..."

    printf "%-8s | %-15s | %-20s | %-20s | %-15s | %-50s\n" "PID" "Environment" "Influence Mode" "Start Time" "Instance ID" "Log File"
    printf "%-8s | %-15s | %-20s | %-20s | %-15s | %-50s\n" "--------" "---------------" "--------------------" "--------------------" "---------------" "--------------------------------------------------"

    for PID_FILE in train_pids_*.txt; do
        [[ -e "$PID_FILE" ]] || continue  # Skip if no PID files exist

        while read -r PID ENVIRONMENT_TYPE INFLUENCE_MODE START_TIME INSTANCE_ID LOG_FILE; do
            if ps -p "$PID" > /dev/null 2>&1; then
                printf "%-8s | %-15s | %-20s | %-20s | %-15s | %-50s\n" "$PID" "$ENVIRONMENT_TYPE" "$INFLUENCE_MODE" "$START_TIME" "$INSTANCE_ID" "$LOG_FILE"
            fi
        done < "$PID_FILE"
    done
}


# Function to query full details of a specific PID
query_job() {
    if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
        echo "Usage: $0 query <environment_type> <influence_mode> <PID>"
        exit 1
    fi

    ENVIRONMENT_TYPE="$1"
    INFLUENCE_MODE="$2"
    PID_TO_QUERY="$3"
    PID_FILE="train_pids_${ENVIRONMENT_TYPE}_${INFLUENCE_MODE}.txt"

    JOB_DETAILS=$(grep "^$PID_TO_QUERY " "$PID_FILE")

    if [[ -z "$JOB_DETAILS" ]]; then
        echo "No matching PID found for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE'."
        exit 1
    fi

    IFS=' ' read -r PID ENVIRONMENT_TYPE INFLUENCE_MODE START_TIME INSTANCE_ID LOG_FILE <<< "$JOB_DETAILS"
    echo "Full details for PID $PID for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE':"
    echo "Start Time:  $START_TIME"
    echo "Instance ID: $INSTANCE_ID"
    echo "Log File:    $LOG_FILE"
}

# Function to kill a specific job
kill_job() {
    if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
        echo "Usage: $0 kill <environment_type> <influence_mode> <PID>"
        exit 1
    fi

    ENVIRONMENT_TYPE="$1"
    INFLUENCE_MODE="$2"
    PID_TO_KILL="$3"
    PID_FILE="train_pids_${ENVIRONMENT_TYPE}_${INFLUENCE_MODE}.txt"

    if ! grep -q "^$PID_TO_KILL " "$PID_FILE"; then
        echo "No matching PID found for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE'."
        exit 1
    fi

    kill "$PID_TO_KILL" && echo "Killed process $PID_TO_KILL for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE'."
    sed -i "/^$PID_TO_KILL /d" "$PID_FILE"
}

# Function to stop all jobs for the given <environment_type> <influence_mode>
stop_all_jobs() {
    if [[ -z "$1" || -z "$2" ]]; then
        echo "Usage: $0 stop <environment_type> <influence_mode>"
        exit 1
    fi

    ENVIRONMENT_TYPE="$1"
    INFLUENCE_MODE="$2"
    PID_FILE="train_pids_${ENVIRONMENT_TYPE}_${INFLUENCE_MODE}.txt"

    if [[ ! -f "$PID_FILE" || ! -s "$PID_FILE" ]]; then
        echo "No running jobs to stop for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE'."
        exit 0
    fi

    echo "Stopping all training jobs for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE'..."
    while read -r PID _; do
        kill "$PID" 2>/dev/null
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    echo "All training jobs for '$ENVIRONMENT_TYPE' '$INFLUENCE_MODE' have been terminated."
}

# Function to list all available variable combinations
list_experiments() {
    echo "Available experiments:"
    ls train_pids_*.txt 2>/dev/null | sed 's/train_pids_\(.*\)\.txt/\1/' || echo "No experiments found."
}

# Function to archive logs
clean_logs() {
    ARCHIVE_DIR="logs/archive"
    mkdir -p "$ARCHIVE_DIR"

    # Move all regular files from logs/ to logs/archive/
    find logs/ -maxdepth 1 -type f -exec mv {} "$ARCHIVE_DIR"/ \;

    echo "All log files have been moved to '$ARCHIVE_DIR'."
}

# Function to display help information
show_help() {
    echo "Usage: $0 {start|list|list_all|query|kill|stop|list_experiments|clean_logs|help} [arguments]"
    echo ""
    echo "Commands:"
    echo "  start <environment_type> <influence_mode>            Start multiple training jobs for the given environment and influence combination."
    echo "  list <environment_type> <influence_mode>             Show running jobs for the specified environment and influence combination."
    echo "  query <environment_type> <influence_mode> <PID>        Show full details for a specific job in the specified environment and influence combination."
    echo "  kill <environment_type> <influence_mode> <PID>         Terminate a specific job for the specified environment and influence combination."
    echo "  stop <environment_type> <influence_mode>             Terminate all running jobs for the specified environment and influence combination."
    echo "  list_experiments                           List all available environment and influence combinations."
    echo "  list_all                           List all running jobs."
    echo "  clean_logs                              Move the current logs into an archive subdirectory in logs/archive"
    echo "  help                                       Display this help message."
    echo ""
    echo "Examples:"
    echo "  $0 start foo bar"
    echo "  $0 list foo bar"
    echo "  $0 query foo bar 12345"
    echo "  $0 kill foo bar 12345"
    echo "  $0 stop foo bar"
    echo "  $0 list_experiments"
}

# Parse command-line arguments
case "$1" in
    start)
        start_jobs "$2" "$3"
        ;;
    list)
        list_jobs "$2" "$3"
        ;;
    list_all)
        list_all_jobs
        ;;
    query)
        query_job "$2" "$3" "$4"
        ;;
    kill)
        kill_job "$2" "$3" "$4"
        ;;
    stop)
        stop_all_jobs "$2" "$3"
        ;;
    list_experiments)
        list_experiments
        ;;
    help)
        show_help
        ;;
    clean_logs)
        clean_logs
        ;;
    *)
        echo "Invalid command. Use '$0 help' for usage details."
        exit 1
        ;;
esac