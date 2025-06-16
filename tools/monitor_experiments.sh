#!/bin/bash
# E-D3DGS Experiment Monitor
# Easy way to track your running and completed experiments

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo "🔍 E-D3DGS Experiment Monitor"
    echo "============================="
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --status, -s        Show all experiment statuses"
    echo "  --logs, -l [JOB]    Show logs for specific job (or latest if no job specified)"
    echo "  --progress, -p      Show training progress of all running experiments"
    echo "  --training, -t      Show detailed training metrics (recent 5 jobs)
  --all, -a           Show all jobs (use with --training)"
    echo "  --clean, -c         Clean up old completed experiment files"
    echo "  --watch, -w [JOB]   Watch logs in real-time for specific job"
    echo "  --list, -ls         List all experiment files organized by dataset"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --status                    # Show all experiment statuses"
    echo "  $0 --logs 12360               # Show logs for job 12360"
    echo "  $0 --training                 # Show training progress for all running jobs"
    echo "  $0 --watch                    # Watch latest experiment logs"
    echo "  $0 --progress                 # Show progress of running experiments"
}

# Function to get job status with colors
get_job_status() {
    local job_id=$1
    local status=$(squeue -j $job_id -h -o "%T" 2>/dev/null)
    
    case $status in
        "RUNNING")
            echo -e "${GREEN}RUNNING${NC}"
            ;;
        "PENDING")
            echo -e "${YELLOW}PENDING${NC}"
            ;;
        "COMPLETED")
            echo -e "${BLUE}COMPLETED${NC}"
            ;;
        "FAILED")
            echo -e "${RED}FAILED${NC}"
            ;;
        "CANCELLED")
            echo -e "${PURPLE}CANCELLED${NC}"
            ;;
        *)
            # Check if job completed by looking at sacct
            local sacct_status=$(sacct -j $job_id -n -o State 2>/dev/null | head -1 | tr -d ' ')
            case $sacct_status in
                "COMPLETED")
                    echo -e "${BLUE}COMPLETED${NC}"
                    ;;
                "FAILED")
                    echo -e "${RED}FAILED${NC}"
                    ;;
                "CANCELLED")
                    echo -e "${PURPLE}CANCELLED${NC}"
                    ;;
                *)
                    echo -e "${RED}UNKNOWN${NC}"
                    ;;
            esac
            ;;
    esac
}

# Function to extract training progress from error logs
get_training_progress() {
    local job_id=$1
    local err_file=$(find experiments/slurm_logs -name "*_${job_id}.err" 2>/dev/null)
    
    if [ -z "$err_file" ] || [ ! -f "$err_file" ]; then
        echo "No training data"
        return
    fi
    
    # Get the latest training progress line
    local latest_progress=$(grep "Training progress:" "$err_file" | tail -1)
    
    if [ -n "$latest_progress" ]; then
        # Extract iteration, loss, psnr, and points
        local iteration=$(echo "$latest_progress" | grep -o '[0-9]\+/80000' | head -1)
        local loss=$(echo "$latest_progress" | grep -o 'Loss=[0-9]\+\.[0-9]\+' | cut -d'=' -f2)
        local psnr=$(echo "$latest_progress" | grep -o 'psnr=[0-9]\+\.[0-9]\+' | cut -d'=' -f2)
        local points=$(echo "$latest_progress" | grep -o 'point=[0-9]\+' | cut -d'=' -f2)
        
        if [ -n "$iteration" ]; then
            local current_iter=$(echo "$iteration" | cut -d'/' -f1)
            local total_iter=$(echo "$iteration" | cut -d'/' -f2)
            local percent=$(echo "scale=1; $current_iter * 100 / $total_iter" | bc -l 2>/dev/null || echo "0.0")
            
            echo "${iteration} (${percent}%) | Loss: ${loss:-N/A} | PSNR: ${psnr:-N/A} | Points: ${points:-N/A}"
        else
            echo "Initializing..."
        fi
    else
        # Check if still in initialization phase
        local init_lines=$(grep -c "Loading\|Reading\|Number of points" "$err_file" 2>/dev/null)
        if [ "$init_lines" -gt 0 ]; then
            echo "Initializing..."
        else
            echo "No training data"
        fi
    fi
}

# Function to show detailed training metrics
show_training_metrics() {
    echo -e "${CYAN}🚂 Training Progress Monitor${NC}"
    echo "============================"
    
    # Get all running jobs
    local running_jobs=$(squeue -u $USER -h -o "%i %j" 2>/dev/null)
    
    if [ -z "$running_jobs" ]; then
        echo "No jobs currently running"
        return
    fi
    
    echo -e "\nJob ID    Experiment                        Progress                                    Status"
    echo "------------------------------------------------------------------------------------------------"
    
    echo "$running_jobs" | while read job_id job_name; do
        local progress=$(get_training_progress "$job_id")
        local status=$(get_job_status "$job_id")
        
        # Extract experiment name from job name if possible
        local exp_name="$job_name"
        if [[ "$job_name" == *"_"* ]]; then
            exp_name=$(echo "$job_name" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')
        fi
        
        printf "%-8s  %-30s  %-40s  %s\n" "$job_id" "$exp_name" "$progress" "$status"
    done
}

# Function to show experiment status
show_status() {
    echo -e "${CYAN}🔍 E-D3DGS Experiment Status${NC}"
    echo "================================"
    
    # Show currently running jobs
    echo -e "\n${YELLOW}📊 SLURM Jobs:${NC}"
    local running_jobs=$(squeue -u $USER -h -o "%i %P %j %T %M %R" 2>/dev/null)
    if [ -n "$running_jobs" ]; then
        echo "Job ID    Partition    Name                          Status    Time     Node"
        echo "------------------------------------------------------------------------"
        echo "$running_jobs" | while read line; do
            echo "$line"
        done
    else
        echo "No jobs currently in SLURM queue"
    fi
    
    # Show recent completed jobs
    echo -e "\n${YELLOW}📋 Recent Jobs (last 24h):${NC}"
    local recent_jobs=$(sacct -u $USER -S $(date -d '1 day ago' +%Y-%m-%d) -o JobID,JobName,State,ExitCode,Start,End --parsable2 2>/dev/null | tail -n +2)
    if [ -n "$recent_jobs" ]; then
        echo "Job ID    Name                          Status      Exit  Start Time       End Time"
        echo "---------------------------------------------------------------------------------"
        echo "$recent_jobs" | while IFS='|' read jobid name state exit start end; do
            printf "%-8s  %-28s  %-10s  %-4s  %-15s  %s\n" "$jobid" "$name" "$state" "$exit" "$start" "$end"
        done
    else
        echo "No recent jobs found"
    fi
}

# Function to show experiment progress
show_progress() {
    echo -e "${CYAN}📈 Experiment Progress${NC}"
    echo "======================"
    
    # First show training progress for running jobs
    local running_jobs=$(squeue -u $USER -h -o "%i" 2>/dev/null)
    
    if [ -n "$running_jobs" ]; then
        echo -e "\n${YELLOW}🚂 Active Training Progress:${NC}"
        echo "Job ID    Progress                                    Metrics"
        echo "------------------------------------------------------------------------"
        
        echo "$running_jobs" | while read job_id; do
            local progress=$(get_training_progress "$job_id")
            printf "%-8s  %s\n" "$job_id" "$progress"
        done
    fi
    
    # Then show stage-based progress files if they exist
    local progress_files=$(find experiments/slurm_logs -name "*_progress.txt" 2>/dev/null)
    
    if [ -n "$progress_files" ]; then
        echo -e "\n${YELLOW}📋 Stage Progress:${NC}"
        echo "Experiment                                    Stage        Time"
        echo "----------------------------------------------------------------"
        
        for file in $progress_files; do
            local basename=$(basename "$file" "_progress.txt")
            local latest_stage=$(tail -1 "$file" 2>/dev/null)
            
            if [ -n "$latest_stage" ]; then
                local stage=$(echo "$latest_stage" | cut -d'|' -f1)
                local time=$(echo "$latest_stage" | cut -d'|' -f2)
                
                # Color code the stage
                case $stage in
                    "STARTED")
                        stage_colored="${YELLOW}STARTED${NC}"
                        ;;
                    "TRAINING")
                        stage_colored="${BLUE}TRAINING${NC}"
                        ;;
                    "TRAINING_DONE")
                        stage_colored="${GREEN}TRAINING_DONE${NC}"
                        ;;
                    "RENDERING")
                        stage_colored="${PURPLE}RENDERING${NC}"
                        ;;
                    "RENDERING_DONE")
                        stage_colored="${GREEN}RENDERING_DONE${NC}"
                        ;;
                    "EVALUATION")
                        stage_colored="${CYAN}EVALUATION${NC}"
                        ;;
                    "COMPLETED")
                        stage_colored="${GREEN}COMPLETED${NC}"
                        ;;
                    "*_FAILED")
                        stage_colored="${RED}${stage}${NC}"
                        ;;
                    *)
                        stage_colored="$stage"
                        ;;
                esac
                
                printf "%-40s  %-20s  %s\n" "$basename" "$stage_colored" "$time"
            fi
        done
    fi
    
    if [ -z "$running_jobs" ] && [ -z "$progress_files" ]; then
        echo "No active experiments found"
    fi
}

# Function to show logs (now prioritizes error logs for training progress)
show_logs() {
    local job_id=$1
    
    if [ -z "$job_id" ]; then
        # Find the most recent error log file (contains training progress)
        local latest_err_log=$(find experiments/slurm_logs -name "*.err" -type f -exec ls -t {} + 2>/dev/null | head -1)
        local latest_out_log=$(find experiments/slurm_logs -name "*.out" -type f -exec ls -t {} + 2>/dev/null | head -1)
        
        if [ -n "$latest_err_log" ]; then
            echo -e "${CYAN}📄 Latest training progress (error log):${NC} $latest_err_log"
            echo "================================"
            # Show last 30 lines of training progress
            grep "Training progress:" "$latest_err_log" | tail -10
            echo ""
            echo -e "${YELLOW}Last 20 lines of error log:${NC}"
            tail -20 "$latest_err_log"
        elif [ -n "$latest_out_log" ]; then
            echo -e "${CYAN}📄 Latest experiment log:${NC} $latest_out_log"
            echo "================================"
            tail -50 "$latest_out_log"
        else
            echo "No log files found"
            return
        fi
    else
        # Find log files for specific job (prioritize error log)
        local err_file=$(find experiments/slurm_logs -name "*_${job_id}.err" 2>/dev/null)
        local out_file=$(find experiments/slurm_logs -name "*_${job_id}.out" 2>/dev/null)
        
        if [ -n "$err_file" ]; then
            echo -e "${CYAN}📄 Training progress for job $job_id:${NC} $err_file"
            echo "================================"
            # Show training progress summary
            local progress=$(get_training_progress "$job_id")
            echo -e "${YELLOW}Current Progress:${NC} $progress"
            echo ""
            echo -e "${YELLOW}Recent Training Progress:${NC}"
            grep "Training progress:" "$err_file" | tail -10
            echo ""
            echo -e "${YELLOW}Last 20 lines of error log:${NC}"
            tail -20 "$err_file"
        elif [ -n "$out_file" ]; then
            echo -e "${CYAN}📄 Output log for job $job_id:${NC} $out_file"
            echo "================================"
            tail -50 "$out_file"
        else
            echo "No log files found for job $job_id"
            return
        fi
    fi
}

# Function to watch logs in real-time (now prioritizes error logs)
watch_logs() {
    local job_id=$1
    
    if [ -z "$job_id" ]; then
        # Find the most recent error log file
        local latest_err_log=$(find experiments/slurm_logs -name "*.err" -type f -exec ls -t {} + 2>/dev/null | head -1)
        local latest_out_log=$(find experiments/slurm_logs -name "*.out" -type f -exec ls -t {} + 2>/dev/null | head -1)
        
        if [ -n "$latest_err_log" ]; then
            echo -e "${CYAN}👀 Watching latest training progress:${NC} $latest_err_log"
            echo "Press Ctrl+C to stop watching"
            echo "================================"
            tail -f "$latest_err_log"
        elif [ -n "$latest_out_log" ]; then
            echo -e "${CYAN}👀 Watching latest experiment log:${NC} $latest_out_log"
            echo "Press Ctrl+C to stop watching"
            echo "================================"
            tail -f "$latest_out_log"
        else
            echo "No log files found"
            return
        fi
    else
        # Find log files for specific job (prioritize error log)
        local err_file=$(find experiments/slurm_logs -name "*_${job_id}.err" 2>/dev/null)
        local out_file=$(find experiments/slurm_logs -name "*_${job_id}.out" 2>/dev/null)
        
        if [ -n "$err_file" ]; then
            echo -e "${CYAN}👀 Watching training progress for job $job_id:${NC} $err_file"
            echo "Press Ctrl+C to stop watching"
            echo "================================"
            tail -f "$err_file"
        elif [ -n "$out_file" ]; then
            echo -e "${CYAN}👀 Watching job $job_id:${NC} $out_file"
            echo "Press Ctrl+C to stop watching"
            echo "================================"
            tail -f "$out_file"
        else
            echo "No log files found for job $job_id"
            return
        fi
    fi
}

# Function to list experiment files
list_experiments() {
    echo -e "${CYAN}📁 Experiment Files Organization${NC}"
    echo "================================="
    
    echo -e "\n${YELLOW}SLURM Scripts:${NC}"
    if [ -d "experiments/slurm_jobs" ]; then
        find experiments/slurm_jobs -name "*.sh" | sort | while read file; do
            local size=$(ls -lh "$file" | awk '{print $5}')
            local date=$(ls -l "$file" | awk '{print $6, $7, $8}')
            echo "  $file ($size, $date)"
        done
    else
        echo "  No SLURM scripts found"
    fi
    
    echo -e "\n${YELLOW}Log Files:${NC}"
    if [ -d "experiments/slurm_logs" ]; then
        find experiments/slurm_logs -name "*.out" -o -name "*.err" | sort | while read file; do
            local size=$(ls -lh "$file" | awk '{print $5}')
            local date=$(ls -l "$file" | awk '{print $6, $7, $8}')
            echo "  $file ($size, $date)"
        done
    else
        echo "  No log files found"
    fi
    
    echo -e "\n${YELLOW}Progress Files:${NC}"
    if [ -d "experiments/slurm_logs" ]; then
        find experiments/slurm_logs -name "*_progress.txt" | sort | while read file; do
            local latest=$(tail -1 "$file" 2>/dev/null | cut -d'|' -f1)
            echo "  $file (latest: $latest)"
        done
    else
        echo "  No progress files found"
    fi
}

# Function to clean up old files
clean_experiments() {
    echo -e "${CYAN}🧹 Cleaning Up Old Experiment Files${NC}"
    echo "===================================="
    
    read -p "This will remove SLURM scripts and logs older than 7 days. Continue? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo "Cleaning up files older than 7 days..."
        
        # Clean old SLURM scripts
        local old_scripts=$(find experiments/slurm_jobs -name "*.sh" -mtime +7 2>/dev/null)
        if [ -n "$old_scripts" ]; then
            echo "$old_scripts" | wc -l | xargs echo "Removing SLURM scripts:"
            echo "$old_scripts" | xargs rm -f
        fi
        
        # Clean old log files
        local old_logs=$(find experiments/slurm_logs -name "*.out" -o -name "*.err" -mtime +7 2>/dev/null)
        if [ -n "$old_logs" ]; then
            echo "$old_logs" | wc -l | xargs echo "Removing log files:"
            echo "$old_logs" | xargs rm -f
        fi
        
        # Clean old progress files
        local old_progress=$(find experiments/slurm_logs -name "*_progress.txt" -mtime +7 2>/dev/null)
        if [ -n "$old_progress" ]; then
            echo "$old_progress" | wc -l | xargs echo "Removing progress files:"
            echo "$old_progress" | xargs rm -f
        fi
        
        echo -e "${GREEN}✅ Cleanup completed!${NC}"
    else
        echo "Cleanup cancelled"
    fi
}

# Function to show training progress from error logs
show_training() {
    local show_all=${1:-false}
    echo -e "${CYAN}🚂 Training Progress Monitor${NC}"
    echo "============================"
    
    # Get currently running job IDs
    local running_jobs=$(squeue -u $USER -h -o "%i" 2>/dev/null | tr '\n' '|' | sed 's/|$//')
    
    # Find error logs from today, prioritizing running jobs
    if [ "$show_all" = "true" ]; then
        local error_logs=$(find experiments/slurm_logs -name "*.err" -newermt "$(date +%Y-%m-%d)" 2>/dev/null | sort -t_ -k4 -nr)
        echo "📋 Showing all jobs from today"
    else
        local error_logs=$(find experiments/slurm_logs -name "*.err" -newermt "$(date +%Y-%m-%d)" 2>/dev/null | sort -t_ -k4 -nr | head -10)
        echo "📋 Showing recent 5 jobs (use --all to see all)"
    fi
    
    if [ -z "$error_logs" ]; then
        echo "❌ No training logs found for today"
        return 1
    fi
    
    local jobs_shown=0
    for log_file in $error_logs; do
        local job_id=$(basename "$log_file" | grep -o '[0-9]\+' | tail -1)
        local scene_name=$(basename "$log_file" | sed 's/_[0-9]\+\.err$//' | sed 's/.*_//')
        
        # Check if this job is currently running
        local is_running=""
        if echo "$running_jobs" | grep -q "$job_id"; then
            is_running=" 🟢 RUNNING"
        else
            is_running=" 🔴 COMPLETED/FAILED"
        fi
        
        echo -e "\n📊 Job $job_id ($scene_name)$is_running:"
        echo "----------------------------------------"
        
        jobs_shown=$((jobs_shown + 1))
        if [ "$show_all" != "true" ] && [ $jobs_shown -ge 5 ]; then
            echo -e "\n... (showing only 5 most recent jobs, use --all for complete list)"
            break
        fi
        
        # Get the latest training progress line
        local latest_line=$(tail -50 "$log_file" 2>/dev/null | grep "Training progress:" | tail -1)
        
        if [ -n "$latest_line" ]; then
            # Extract values using pattern matching for the actual format: "Training progress:   7%|▋         | 5290/80000"
            local iteration=$(echo "$latest_line" | sed -n 's/.*|\s*\([0-9]\+\)\/\([0-9]\+\).*/\1/p')
            local total=$(echo "$latest_line" | sed -n 's/.*|\s*\([0-9]\+\)\/\([0-9]\+\).*/\2/p')
            local loss=$(echo "$latest_line" | sed -n 's/.*Loss=\([0-9.]\+\).*/\1/p')
            local psnr=$(echo "$latest_line" | sed -n 's/.*psnr=\([0-9.]\+\).*/\1/p')
            local points=$(echo "$latest_line" | sed -n 's/.*point=\([0-9]\+\).*/\1/p')
            
            if [ -n "$iteration" ] && [ -n "$total" ]; then
                local percent=$((iteration * 100 / total))
                
                echo "  📈 Progress: $iteration/$total ($percent%)"
                echo "  📉 Loss: ${loss:-N/A}"
                echo "  🎯 PSNR: ${psnr:-N/A}"
                echo "  🔵 Points: ${points:-N/A}"
                
                # Show last few PSNR values for trend
                echo "  📊 Recent PSNR trend:"
                tail -100 "$log_file" 2>/dev/null | grep "psnr=" | tail -3 | while read line; do
                    local iter=$(echo "$line" | sed -n 's/.*|\s*\([0-9]\+\)\/\([0-9]\+\).*/\1/p')
                    local psnr_val=$(echo "$line" | sed -n 's/.*psnr=\([0-9.]\+\).*/\1/p')
                    if [ -n "$iter" ] && [ -n "$psnr_val" ]; then
                        echo "     [$iter] $psnr_val"
                    fi
                done
            else
                echo "  ⏳ Parsing training progress..."
            fi
        else
            # Check if job is still initializing
            local init_check=$(tail -20 "$log_file" 2>/dev/null | grep -E "Loading|Reading|Number of points|Optimizing")
            if [ -n "$init_check" ]; then
                echo "  🔄 Initializing training..."
            else
                echo "  ❓ No training progress found"
            fi
        fi
        
        # Check job status
        if squeue -u $USER -j $job_id &>/dev/null 2>&1; then
            echo "  ✅ Status: RUNNING"
        else
            echo "  ❌ Status: STOPPED"
        fi
    done
}

# Main script logic
case "${1:-}" in
    --status|-s)
        show_status
        ;;
    --logs|-l)
        show_logs "$2"
        ;;
    --progress|-p)
        show_progress
        ;;
    --training|-t)
        if [ "$2" = "--all" ] || [ "$2" = "-a" ]; then
            show_training true
        else
            show_training false
        fi
        ;;
    --all|-a)
        if [ "$2" = "--training" ] || [ "$2" = "-t" ]; then
            show_training true
        else
            echo "Error: --all must be used with --training"
            show_usage
            exit 1
        fi
        ;;
    --watch|-w)
        watch_logs "$2"
        ;;
    --list|-ls)
        list_experiments
        ;;
    --clean|-c)
        clean_experiments
        ;;
    --help|-h|"")
        show_usage
        ;;
    *)
        echo "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 