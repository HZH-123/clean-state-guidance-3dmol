#!/bin/bash
# Check progress of full_best_config_fast run

RESULT_DIR="experiments_multi/full_best_config"
LOG_FILE="full_best_fast.log"

while true; do
    clear
    echo "=========================================="
    echo "FULL RUN PROGRESS CHECK"
    echo "=========================================="
    echo ""

    # Count completed pockets
    if [ -d "$RESULT_DIR" ]; then
        COMPLETED=$(ls "$RESULT_DIR"/result_*.pt 2>/dev/null | wc -l)
        echo "Completed pockets: $COMPLETED / 100"
        echo "Progress: $(($COMPLETED))%"

        # Progress bar
        printf "["
        for i in $(seq 1 50); do
            if [ $i -le $(($COMPLETED / 2)) ]; then
                printf "="
            else
                printf " "
            fi
        done
        printf "]\n"
    else
        echo "Result directory not found yet"
        COMPLETED=0
    fi

    echo ""
    echo "=========================================="
    echo "GPU STATUS"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU $line"
    done

    echo ""
    echo "=========================================="
    echo "RUNNING PROCESSES"
    echo "=========================================="
    ps aux | grep "sample_multi_guided_diffusion" | grep -v grep | awk '{printf "  PID %s: %s (Running: %s)\n", $2, $14, $10}'

    echo ""
    echo "=========================================="
    echo "RECENT LOG (last 5 lines)"
    echo "=========================================="
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE"
    else
        echo "  Log file not found"
    fi

    echo ""
    echo "=========================================="
    echo "Refresh: $(date '+%H:%M:%S') | Press Ctrl+C to exit"
    echo "=========================================="

    sleep 10
done
