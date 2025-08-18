#!/bin/bash
# Script to find and clean up RAM memory leaks from llama.cpp

echo "🔍 Finding RAM memory leaks from llama.cpp"
echo "==========================================="

# Show current memory usage
echo -e "\n📊 Current Memory Status:"
free -h
echo ""
echo "Top memory consumers:"
ps aux --sort=-%mem | head -10

# Check for memory-mapped files
echo -e "\n📁 Memory-mapped GGUF files:"
lsof | grep -E "\.gguf" | head -20

# Check for deleted but still open files (common cause of memory leaks)
echo -e "\n🗑️ Deleted files still in memory:"
lsof | grep deleted | grep -E "(gguf|model|llama|qwen)" | head -20

# Check shared memory segments
echo -e "\n💾 Shared memory segments:"
ipcs -m

# Function to clean up memory leaks
cleanup_memory() {
    echo -e "\n🧹 Starting memory cleanup..."
    
    # 1. Find and kill processes holding GGUF files
    echo "Looking for processes with GGUF files in memory..."
    for pid in $(lsof | grep "\.gguf" | awk '{print $2}' | sort -u); do
        if [ -n "$pid" ]; then
            echo "Found process $pid holding GGUF file"
            ps -p $pid -o comm,args 2>/dev/null
            kill -TERM $pid 2>/dev/null
            sleep 1
            kill -KILL $pid 2>/dev/null
        fi
    done
    
    # 2. Clear deleted but open files
    echo -e "\nClearing deleted file handles..."
    for pid in $(lsof | grep deleted | grep -E "(gguf|model|llama)" | awk '{print $2}' | sort -u); do
        if [ -n "$pid" ]; then
            echo "Killing process $pid with deleted files"
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # 3. Clear page cache and memory buffers
    echo -e "\nClearing system caches (requires sudo)..."
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 && echo "✅ Caches cleared" || echo "⚠️ Could not clear caches (need sudo)"
    
    # 4. Remove shared memory segments
    echo -e "\nCleaning shared memory segments..."
    for shmid in $(ipcs -m | grep -E "0x00000000|abandoned" | awk '{print $2}' | grep -E "^[0-9]+$"); do
        echo "Removing shared memory segment $shmid"
        ipcrm -m $shmid 2>/dev/null
    done
    
    # 5. Force garbage collection in any running Python processes
    echo -e "\nTriggering Python garbage collection..."
    for pid in $(pgrep python); do
        echo "Sending SIGUSR1 to Python process $pid for GC"
        kill -USR1 $pid 2>/dev/null || true
    done
    
    echo -e "\n✅ Memory cleanup complete"
}

# Function to find memory leaks
find_memory_leaks() {
    echo -e "\n🔎 Detailed memory analysis..."
    
    # Check for large anonymous memory regions (typical for mmap'd models)
    echo -e "\n📍 Large anonymous memory regions (possible leaked models):"
    for pid in $(ps aux | grep -E "python|llama" | grep -v grep | awk '{print $2}'); do
        if [ -d /proc/$pid ]; then
            echo "Process $pid ($(ps -p $pid -o comm= 2>/dev/null)):"
            cat /proc/$pid/maps 2>/dev/null | grep -E "anon|deleted" | \
                awk '{split($1,a,"-"); start=strtonum("0x"a[1]); end=strtonum("0x"a[2]); size=(end-start)/1024/1024; if(size>100) printf "  %.0f MB: %s\n", size, $0}' | \
                head -5
        fi
    done
    
    # Check memory info for specific processes
    echo -e "\n📊 Memory details for Python processes:"
    for pid in $(pgrep python); do
        if [ -f /proc/$pid/status ]; then
            echo "PID $pid:"
            grep -E "VmSize|VmRSS|VmData|VmLib" /proc/$pid/status 2>/dev/null | sed 's/^/  /'
        fi
    done
}

# Menu
echo -e "\n🔧 Memory Cleanup Options:"
echo "1) Show current memory status"
echo "2) Find memory leaks"
echo "3) Perform memory cleanup"
echo "4) Emergency cleanup (aggressive)"
echo "5) Exit"

read -p "Choose option (1-5): " choice

case $choice in
    1)
        free -h
        echo -e "\n📊 Detailed memory info:"
        cat /proc/meminfo | grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached|AnonPages|Mapped|Shmem):"
        ;;
    2)
        find_memory_leaks
        ;;
    3)
        cleanup_memory
        ;;
    4)
        echo "⚠️  WARNING: This will aggressively clean memory!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            # Kill everything Python/llama related
            pkill -9 -f python
            pkill -9 -f llama
            sleep 1
            sync
            echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
            echo "✅ Emergency cleanup done"
        fi
        ;;
    5)
        exit 0
        ;;
    *)
        echo "Invalid option"
        ;;
esac

# Show final status
echo -e "\n📊 Final Memory Status:"
free -h