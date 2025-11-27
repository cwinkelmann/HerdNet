#!/bin/bash

# Parallel training script with staggered starts
# Usage: ./parallel_training.sh

# Exit on any error
set -e

# Setup environment
export PYTHONPATH=$PYTHONPATH:../

# Define experiment configs
configs=(
#    "customdla34_publication_setting"
#    "timmdla34_publication_setting"
#    "timmdla102x_publication_setting"
    "timmdla102_publication_setting"
    "dinoV2_base_publication_setting"
    "dinoV2_small_publication_setting"
    "dinoV2_large_publication_setting"
    # Add more configs here as needed
)

config_path="../configs/experiment_publication_reproduction"
DELAY_BETWEEN_STARTS=30  # seconds

# Create logs directory
if ! mkdir -p logs; then
    echo "ERROR: Failed to create logs directory"
    exit 1
fi

# Store background process IDs
pids=()

# Signal handler for clean shutdown
cleanup() {
    echo ""
    echo "Received interrupt signal. Cleaning up..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating process $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "Cleanup completed."
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Function to sanitize config name for filename
sanitize_filename() {
    local config=$1
    echo "$config" | sed 's/[\/]/_/g'
}

# Function to run a single experiment
run_experiment() {
    local config=$1
    local timestamp=$(date "+%Y%m%d_%H%M%S")
    local safe_config=$(sanitize_filename "$config")
    local logfile="logs/${safe_config}_${timestamp}.log"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting: $config"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Log file: $logfile"

    # Run the experiment and capture output
    if python train_cli.py --config-path="$config_path" --config-name="$config" > "$logfile" 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $config"
        return 0
    else
        local exit_code=$?
        echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $config (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to check if all processes are done
check_completion() {
    local running=0
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            running=$((running + 1))
        fi
    done
    echo $running
}

echo "========================================"
echo "Starting parallel training experiments"
echo "Total configs: ${#configs[@]}"
echo "Delay between starts: ${DELAY_BETWEEN_STARTS} seconds"
echo "========================================"

# Start each experiment with staggered delay
for i in "${!configs[@]}"; do
    config="${configs[$i]}"

    # Start experiment in background
    run_experiment "$config" &

    # Store the PID
    pids+=($!)

    echo "Started $config (PID: $!)"

    # Wait before starting next (except for last one)
    if [ $i -lt $((${#configs[@]} - 1)) ]; then
        echo "Waiting ${DELAY_BETWEEN_STARTS} seconds before next start..."
        sleep $DELAY_BETWEEN_STARTS
    fi
done

echo "========================================"
echo "All experiments started!"
echo "PIDs: ${pids[*]}"
echo "========================================"

# Monitor progress
echo "Monitoring progress... (Ctrl+C to stop monitoring, experiments will continue)"
while true; do
    running=$(check_completion)
    if [ "$running" -eq 0 ]; then
        break
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Still running: $running/${#configs[@]} experiments"
    sleep 30
done

echo "========================================"
echo "All experiments completed!"
echo "Check logs in ./logs/ directory"

# Wait for all background processes and collect exit codes
failed_count=0
successful_configs=()
failed_configs=()

for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    config=${configs[$i]}

    if wait "$pid"; then
        echo "✓ $config completed successfully"
        successful_configs+=("$config")
    else
        echo "✗ $config failed"
        failed_configs+=("$config")
        failed_count=$((failed_count + 1))
    fi
done

echo "========================================"
echo "Summary:"
echo "Total experiments: ${#configs[@]}"
echo "Successful: $((${#configs[@]} - failed_count))"
echo "Failed: $failed_count"

if [ ${#successful_configs[@]} -gt 0 ]; then
    echo ""
    echo "Successful experiments:"
    printf "  - %s\n" "${successful_configs[@]}"
fi

if [ ${#failed_configs[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    printf "  - %s\n" "${failed_configs[@]}"
fi

echo "========================================"

# Exit with error if any experiment failed
if [ $failed_count -gt 0 ]; then
    exit 1
else
    exit 0
fi