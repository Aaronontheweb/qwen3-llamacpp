#!/bin/bash
# Script to optimize memory usage for large model files

echo "🔧 Optimizing system for large model files"
echo "=========================================="

# Function to set up memory optimization
setup_optimization() {
    echo -e "\n📝 Setting memory optimization parameters..."
    
    # 1. Reduce swappiness (prefer keeping apps in RAM over file cache)
    echo "Setting swappiness to 10 (from default 60)..."
    echo 10 | sudo tee /proc/sys/vm/swappiness
    
    # 2. Reduce cache pressure (be more aggressive about reclaiming cache)
    echo "Setting cache pressure to 200 (from default 100)..."
    echo 200 | sudo tee /proc/sys/vm/vfs_cache_pressure
    
    # 3. Set dirty ratio lower (write to disk sooner)
    echo "Setting dirty ratios..."
    echo 5 | sudo tee /proc/sys/vm/dirty_ratio
    echo 10 | sudo tee /proc/sys/vm/dirty_background_ratio
    
    # 4. Create systemd service to clear cache periodically
    cat << 'EOF' | sudo tee /etc/systemd/system/clear-model-cache.service
[Unit]
Description=Clear model file cache periodically
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'sync && echo 1 > /proc/sys/vm/drop_caches'
EOF

    cat << 'EOF' | sudo tee /etc/systemd/system/clear-model-cache.timer
[Unit]
Description=Clear model cache every 30 minutes
Requires=clear-model-cache.service

[Timer]
OnBootSec=30min
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable clear-model-cache.timer
    sudo systemctl start clear-model-cache.timer
    
    echo "✅ Memory optimization configured"
}

# Function to make settings permanent
make_permanent() {
    echo -e "\n📝 Making settings permanent..."
    
    cat << 'EOF' | sudo tee /etc/sysctl.d/99-llama-optimization.conf
# Optimization for large language models
vm.swappiness = 10
vm.vfs_cache_pressure = 200
vm.dirty_ratio = 5
vm.dirty_background_ratio = 10
EOF
    
    sudo sysctl -p /etc/sysctl.d/99-llama-optimization.conf
    echo "✅ Settings saved to /etc/sysctl.d/99-llama-optimization.conf"
}

# Menu
echo -e "\n🔧 Memory Optimization Options:"
echo "1) Show current settings"
echo "2) Apply optimization (temporary)"
echo "3) Apply optimization (permanent)"
echo "4) Disable mmap for models (add to config)"
echo "5) Exit"

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo -e "\nCurrent settings:"
        echo "Swappiness: $(cat /proc/sys/vm/swappiness)"
        echo "Cache pressure: $(cat /proc/sys/vm/vfs_cache_pressure)"
        echo "Dirty ratio: $(cat /proc/sys/vm/dirty_ratio)"
        echo "Dirty background ratio: $(cat /proc/sys/vm/dirty_background_ratio)"
        ;;
    2)
        setup_optimization
        ;;
    3)
        setup_optimization
        make_permanent
        ;;
    4)
        echo -e "\nAdd this to your llama_settings in models_config.json:"
        echo '"use_mmap": false,'
        echo -e "\nThis will load models into RAM directly instead of memory-mapping them."
        ;;
    5)
        exit 0
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo -e "\n📊 Current memory status:"
free -h