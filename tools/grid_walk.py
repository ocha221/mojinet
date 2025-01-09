import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import logging
import os
plt.rcParams['font.family'] = 'Hiragino Sans GB'

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

def grid_walk(image_path, txt_path, is_solo=False, label=None):
    etl_type = None
    if not is_solo:
        for et in ETL_IMAGE_SIZES:
            if et in str(image_path):
                etl_type = et
                break
        
        if not etl_type:
            raise ValueError(f"Unknown ETL type for {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image {image_path}")
    
    if is_solo:
        cell = img
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        
        plt.figure(figsize=(16, 4))
        
        plt.subplot(141, aspect='equal')
        plt.imshow(cell)
        plt.title('Raw Image', fontsize=12)
        
        plt.subplot(142, aspect='equal')
        normalized = cell_gray / 255.0
        plt.imshow(normalized, cmap='gray')
        plt.title('Normalized', fontsize=12)
        
        plt.subplot(143, aspect='equal')
        _, otsu = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(otsu, cmap='gray')
        plt.title("Otsu's Threshold", fontsize=12)
        
        plt.subplot(144, aspect='equal')
        inverted_otsu = cv2.bitwise_not(otsu)
        plt.imshow(inverted_otsu, cmap='gray')
        plt.title('Inverted Otsu', fontsize=12)
        
        plt.suptitle(f"{Path(image_path).name}\nLabel: {label}", fontsize=14)
        plt.tight_layout()
        plt.show()
        return

    #*egular grid walk mode
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
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        
        plt.subplot(441, aspect='equal')
        plt.imshow(cell)
        plt.title('Raw Image', fontsize=12)
        
        plt.subplot(442, aspect='equal')
        normalized = cell_gray / 255.0
        plt.imshow(normalized, cmap='gray')
        plt.title('Normalized', fontsize=12)
        
        plt.subplot(443, aspect='equal')
        _, otsu = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(otsu, cmap='gray')
        plt.title("Otsu's Threshold", fontsize=12)
        
        plt.subplot(444, aspect='equal')
        inverted_otsu = cv2.bitwise_not(otsu)
        plt.imshow(inverted_otsu, cmap='gray')
        plt.title('Inverted Otsu', fontsize=12)
        
        plt.subplot(445)
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
        
        plt.text(0.1, 0.9, label_info + context, fontsize=12, 
                verticalalignment='top', wrap=True)
        
        plt.suptitle(f"Grid Debug Viewer - {Path(image_path).name}\n"
                    f"Use Left/Right arrows to navigate - Position {current_pos}/{len(labels)-1}",
                    fontsize=14)
       

def main():
    parser = argparse.ArgumentParser(description="Walk through the grid (helps find irregularities between grid/labels")
    parser.add_argument('image', help='Path to grid image file')
    parser.add_argument('labels', help='Path to corresponding label text file')
    parser.add_argument('--solo', action='store_true', help='Display a single cell instead of grid walk')
    parser.add_argument('--label', help='Label for solo mode')
    args = parser.parse_args()
    
    grid_walk(args.image, args.labels, args.solo, args.label)

if __name__ == '__main__':
    main()
