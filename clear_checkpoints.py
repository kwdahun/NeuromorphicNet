#!/usr/bin/env python3
"""
Script to backup and clear corrupted checkpoint files.
"""

import os
import shutil
from datetime import datetime

def clear_checkpoints():
    """Backup and clear corrupted checkpoint files."""
    
    checkpoint_dir = r"D:\Workspace\NeuromorphicNet\NeuromorphicNet\checkpoints\MNIST"
    backup_dir = r"D:\Workspace\NeuromorphicNet\NeuromorphicNet\checkpoints\MNIST_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    files_to_backup = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.bin'):
            files_to_backup.append(file)
    
    if not files_to_backup:
        print("No checkpoint files found to clear.")
        return
    
    print(f"Found checkpoint files: {files_to_backup}")
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory: {backup_dir}")
    
    # Backup and remove files
    for file in files_to_backup:
        src_path = os.path.join(checkpoint_dir, file)
        backup_path = os.path.join(backup_dir, file)
        
        # Copy to backup
        shutil.copy2(src_path, backup_path)
        print(f"Backed up: {file}")
        
        # Remove original
        os.remove(src_path)
        print(f"Removed: {file}")
    
    print(f"\n✅ Successfully cleared {len(files_to_backup)} corrupted checkpoint files.")
    print(f"📁 Backup saved to: {backup_dir}")
    print("\nThe training will now start fresh and create new, compatible checkpoint files.")

if __name__ == "__main__":
    clear_checkpoints()