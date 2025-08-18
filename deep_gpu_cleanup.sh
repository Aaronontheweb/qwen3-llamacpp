#!/bin/bash
# Deep GPU memory cleanup script for leaked CUDA memory

echo "🔍 Deep GPU Memory Cleanup"
echo "=========================="

# Show current GPU status
echo -e "\n📊 Current GPU Memory Status:"
nvidia-smi

# Check for any Python processes that might be holding memory
echo -e "\n🐍 All Python processes:"
ps aux | grep python | grep -v grep

# Check CUDA processes
echo -e "\n🎮 NVIDIA processes:"
ps aux | grep nvidia | grep -v grep

# Check for zombie processes
echo -e "\n🧟 Zombie processes:"
ps aux | awk '$8 ~ /^[Zz]/'

# Function to perform deep cleanup
deep_cleanup() {
    echo -e "\n🧹 Starting deep GPU cleanup..."
    
    # 1. Kill ALL Python processes (aggressive!)
    echo "⚠️  Killing all Python processes..."
    pkill -9 python
    pkill -9 python3
    
    # 2. Clear CUDA cache
    echo "Clearing CUDA cache..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # 3. Unload and reload NVIDIA kernel modules (requires root)
    echo -e "\n🔧 Attempting to reset NVIDIA drivers (requires sudo)..."
    
    # Try nvidia-smi reset
    sudo nvidia-smi --gpu-reset 2>/dev/null && echo "✅ GPU reset successful" || {
        echo "⚠️  GPU reset failed, trying module reload..."
        
        # More aggressive: unload/reload kernel modules
        echo "Attempting to reload NVIDIA kernel modules..."
        sudo rmmod nvidia_uvm 2>/dev/null || true
        sudo rmmod nvidia 2>/dev/null || true
        sleep 2
        sudo modprobe nvidia 2>/dev/null || true
        sudo modprobe nvidia_uvm 2>/dev/null || true
    }
    
    # 4. Alternative: Restart display manager (very aggressive!)
    echo -e "\n❓ Do you want to restart the display manager? This will log you out! (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo systemctl restart gdm 2>/dev/null || \
        sudo systemctl restart lightdm 2>/dev/null || \
        sudo systemctl restart sddm 2>/dev/null || \
        echo "Could not restart display manager"
    fi
    
    echo -e "\n✅ Deep cleanup complete"
}

# Function to check what's using GPU memory without visible processes
check_gpu_handles() {
    echo -e "\n🔍 Checking GPU memory handles..."
    
    # Use nvidia-smi to show compute processes
    nvidia-smi pmon -c 1
    
    # Check /proc for any file descriptors to nvidia devices
    echo -e "\n📁 Checking for nvidia device file handles:"
    sudo lsof 2>/dev/null | grep nvidia | head -20
    
    # Check kernel memory
    echo -e "\n💾 Kernel GPU memory info:"
    cat /proc/driver/nvidia/gpus/*/information 2>/dev/null || echo "No permission to read GPU info"
}

# Menu
echo -e "\n🔧 GPU Memory Cleanup Options:"
echo "1) Show detailed GPU memory usage"
echo "2) Check for hidden GPU handles"
echo "3) Perform deep cleanup (kills all Python!)"
echo "4) Exit"

read -p "Choose option (1-4): " choice

case $choice in
    1)
        nvidia-smi
        echo -e "\n📊 Detailed memory info:"
        nvidia-smi -q -d MEMORY
        ;;
    2)
        check_gpu_handles
        ;;
    3)
        echo "⚠️  WARNING: This will kill ALL Python processes!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            deep_cleanup
        fi
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        ;;
esac

# Show final status
echo -e "\n📊 Final GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv