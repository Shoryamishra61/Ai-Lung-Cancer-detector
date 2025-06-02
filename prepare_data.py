import os
import shutil
from pathlib import Path
import random
from PIL import Image
import numpy as np

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dirs = ['train', 'val', 'test']
    categories = ['normal', 'stage0', 'stage1', 'stage2', 'stage3', 'stage4']
    
    for base in base_dirs:
        for category in categories:
            Path(f'data/{base}/{category}').mkdir(parents=True, exist_ok=True)

def process_and_split_images(input_folder, split_ratio=(0.7, 0.15, 0.15)):
    """
    Process images from input folder and split them into train/val/test sets.
    
    Args:
        input_folder: Folder containing category subfolders with images
        split_ratio: (train, val, test) ratio for splitting data
    """
    # Convert to absolute paths
    input_folder = os.path.abspath(input_folder)
    data_dir = os.path.abspath('data')
    
    setup_directories()
    
    categories = ['normal', 'stage0', 'stage1', 'stage2', 'stage3', 'stage4']
    
    for category in categories:
        category_path = os.path.join(input_folder, category)
        if not os.path.exists(category_path):
            print(f"Skipping {category} - directory not found at {category_path}")
            continue
            
        # Get all images in the category
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        
        print(f"Found {len(images)} images in {category}")
        
        if not images:
            print(f"No images found in {category_path}")
            continue
            
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split sizes ensuring at least one image per split
        n_images = len(images)
        if n_images < 3:
            # For very small datasets, distribute as evenly as possible
            n_train = max(1, n_images // 3)
            n_val = max(1, (n_images - n_train) // 2)
            n_test = n_images - n_train - n_val
        else:
            # For larger datasets, use the ratio
            n_train = max(1, int(n_images * split_ratio[0]))
            n_val = max(1, int(n_images * split_ratio[1]))
            n_test = n_images - n_train - n_val
        
        # Split images
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Copy and preprocess images
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, split_imgs in splits.items():
            target_dir = os.path.join(data_dir, split_name, category)
            print(f"Processing {len(split_imgs)} images for {split_name}/{category}")
            
            for img_name in split_imgs:
                src_path = os.path.join(category_path, img_name)
                dst_path = os.path.join(target_dir, img_name)
                
                try:
                    # Open and preprocess image
                    img = Image.open(src_path)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to model input size
                    img = img.resize((224, 224))
                    
                    # Save preprocessed image
                    img.save(dst_path)
                    print(f"Processed and saved: {dst_path}")
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    # Example usage
    input_folder = "raw_images"  # Your folder containing category subfolders
    process_and_split_images(input_folder) 