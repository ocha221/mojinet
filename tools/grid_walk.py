import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ETL_IMAGE_SIZES = {
    'ETL1': (64, 63),
    'ETL2': (60, 60),
    'ETL3': (72, 76),
    'ETL4': (72, 76),
    'ETL5': (72, 76),
    'ETL6': (64, 63),
    'ETL7': (64, 63),
    'ETL8B': (64, 63),
    'ETL8G': (128, 127),
    'ETL9B': (64, 63),
    'ETL9G': (128, 127)
}

def read_labels(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info("Raw content hex:")
                logger.info(' '.join(hex(ord(c)) for c in content))
                
                content = content.replace('\n', '') #* basically files have newlines in them sometimes and it messes up the counter. if you remove the replace() you can see for yourself
                chars = list(content)
                logger.info("\nCharacter breakdown:")
                for i, c in enumerate(chars):
                    logger.info(f"Position {i}: '{c}' (hex: {hex(ord(c))}, unicode name: {repr(c)})")
                
                return chars
    except UnicodeDecodeError:
        pass
    
    raise ValueError(f"Failed to read labels from {txt_path}")

def grid_walk(image_path, txt_path):
    etl_type = None
    for et in ETL_IMAGE_SIZES:
        if et in str(image_path):
            etl_type = et
            break
    
    if not etl_type:
        raise ValueError(f"Unknown ETL type for {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image {image_path}")
    
    labels = read_labels(txt_path)
    cell_width, cell_height = ETL_IMAGE_SIZES[etl_type]
    rows = 40
    cols = 50
    
    plt.style.use('dark_background')
    current_pos = 0
    
    def on_key(event):
        nonlocal current_pos
        if event.key == 'right' and current_pos < len(labels) - 1:
            current_pos += 1
            update_display()
        elif event.key == 'left' and current_pos > 0:
            current_pos -= 1
            update_display()
            
    def update_display():
        plt.clf()
        
        row = current_pos // cols
        col = current_pos % cols
        
        
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        
        cell = img[y1:y2, x1:x2]
        
        
        plt.subplot(121)
        plt.imshow(cell, cmap='gray')
        plt.title(f'Cell Image (Row: {row}, Col: {col})')
        
        
        plt.subplot(122)
        plt.axis('off')
        current_label = labels[current_pos]
        label_info = (
            f"Position: {current_pos}\n"
            f"Label: '{current_label}'\n"
            f"Hex: {hex(ord(current_label))}\n"
            f"Unicode Name: {repr(current_label)}\n\n"
            f"Context (5 chars before/after):\n"
        )
    
        start = max(0, current_pos - 5)
        end = min(len(labels), current_pos + 6)
        context_chars = labels[start:end]
        context = ' '.join(f"[{i+start}:'{c}']" for i, c in enumerate(context_chars))
        
        plt.text(0.1, 0.9, label_info + context, fontsize=10, 
                verticalalignment='top', wrap=True)
        
        plt.suptitle(f"Grid Debug Viewer - {Path(image_path).name}\n"
                    f"Use Left/Right arrows to navigate - Position {current_pos}/{len(labels)-1}")
        plt.tight_layout()
        plt.draw()
    
    
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    
    update_display()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Walk through the grid (helps find irregularities between grid/labels")
    parser.add_argument('image', help='Path to grid image file')
    parser.add_argument('labels', help='Path to corresponding label text file')
    args = parser.parse_args()
    
    grid_walk(args.image, args.labels)

if __name__ == '__main__':
    main()
