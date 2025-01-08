import os
from torchvision.datasets import ImageFolder
import sys
from pathlib import Path

def check_directory(path):
    print(f"\nChecking directory: {path}")
    print("-" * 50)
    

    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return
    
    extensions = {}
    empty_dirs = []
    total_files = 0
    
    for root, dirs, files in os.walk(path):
        rel_path = os.path.relpath(root, path)
        if not files and root != path:
            empty_dirs.append(rel_path)
            
        for file in files:
            ext = os.path.splitext(file.lower())[1]
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
                total_files += 1


    print(f"Total files found: {total_files}")
    print("\nFile extensions found:")
    for ext, count in extensions.items():
        print(f"{ext}: {count} files")
        
    if empty_dirs:
        print("\nEmpty directories found:")
        for d in empty_dirs:
            print(f"- {d}")
    

    first_level = [d for d in os.listdir(path) 
                  if os.path.isdir(os.path.join(path, d))]
    print(f"\nFirst level directories ({len(first_level)}):")
    print(first_level[:5])

path = sys.argv[1]
check_directory(path)

try:
    dataset = ImageFolder(path)
    print(f"\nImageFolder Success:")
    print(f"Images loaded: {len(dataset)}")
    print(f"Classes found: {len(dataset.classes)}")
    print(f"First few classes: {dataset.classes[:5]}")
except Exception as e:
    print(f"\nImageFolder Error: {str(e)}")