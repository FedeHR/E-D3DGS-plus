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
    echo "  --progress, -p      Show progress of all running experiments"
    echo "  --clean, -c         Clean up old completed experiment files"
    echo "  --watch, -w [JOB]   Watch logs in real-time for specific job"
    echo "  --list, -ls         List all experiment files organized by dataset"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --status                    # Show all experiment statuses"
    echo "  $0 --logs 12360               # Show logs for job 12360"
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
    
    # Find all progress files
    local progress_files=$(find experiments/slurm_logs -name "*_progress.txt" 2>/dev/null)
    
    if [ -z "$progress_files" ]; then
        echo "No progress files found"
        return
    fi
    
    echo -e "\nExperiment                                    Stage        Time"
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
}

# Function to show logs
show_logs() {
    local job_id=$1
    
    if [ -z "$job_id" ]; then
        # Find the most recent log file
        local latest_log=$(find experiments/slurm_logs -name "*.out" -type f -exec ls -t {} + 2>/dev/null | head -1)
        if [ -z "$latest_log" ]; then
            echo "No log files found"
            return
        fi
        echo -e "${CYAN}📄 Latest experiment log:${NC} $latest_log"
        echo "================================"
        tail -50 "$latest_log"
    else
        # Find log file for specific job
        local log_file=$(find experiments/slurm_logs -name "*_${job_id}.out" 2>/dev/null)
        if [ -z "$log_file" ]; then
            echo "No log file found for job $job_id"
            return
        fi
        echo -e "${CYAN}📄 Log for job $job_id:${NC} $log_file"
        echo "================================"
        tail -50 "$log_file"
    fi
}

# Function to watch logs in real-time
watch_logs() {
    local job_id=$1
    
    if [ -z "$job_id" ]; then
        # Find the most recent log file
        local latest_log=$(find experiments/slurm_logs -name "*.out" -type f -exec ls -t {} + 2>/dev/null | head -1)
        if [ -z "$latest_log" ]; then
            echo "No log files found"
            return
        fi
        echo -e "${CYAN}👀 Watching latest experiment log:${NC} $latest_log"
        echo "Press Ctrl+C to stop watching"
        echo "================================"
        tail -f "$latest_log"
    else
        # Find log file for specific job
        local log_file=$(find experiments/slurm_logs -name "*_${job_id}.out" 2>/dev/null)
        if [ -z "$log_file" ]; then
            echo "No log file found for job $job_id"
            return
        fi
        echo -e "${CYAN}👀 Watching job $job_id:${NC} $log_file"
        echo "Press Ctrl+C to stop watching"
        echo "================================"
        tail -f "$log_file"
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