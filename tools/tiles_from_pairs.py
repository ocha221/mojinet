import cv2 # type: ignore
import numpy as np
import os
from pathlib import Path
import glob
from multiprocessing import Pool
import itertools
import unicodedata

def jis_to_hiragana(text):
    if not text:
        return None
    x = unicodedata.normalize('NFKC', text).replace('ヴ', 'う゛')
    #* JIS x 0201 mapped files contain half-width katakana (from old terminal intefaces). 
    #* normalising through NFKC will convert half-width katakana to full-width
    #* then we subtract the offset (from UTF docs) to convert to hiragana
    #* we dont care about predicting half-width katakana so this should be fine
    if 0x30A1 <= ord(x) <= 0x30F3:  
        return chr(ord(x) - 0x60)  # Convert to hiragana
    return text

def load_jis_map(filename):  # * done
    jis_to_unicode = {}

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    jis_code = int(parts[0].replace("0x", ""), 16)
                    unicode_value = int(parts[1].replace("0x", ""), 16)
                    jis_to_unicode[jis_code] = unicode_value
    return jis_to_unicode

def get_character(jis_code, mapping):
    """Convert JIS code to Unicode character."""
    if isinstance(jis_code, str):
        jis_code = int(jis_code.replace('0x', ''), 16)
    
    unicode_value = mapping.get(jis_code)
    if unicode_value is None:
        return None
    return chr(unicode_value)

def split_grid_image(image_path, txt_path, output_dir, grid_size=(63,64)):
    try:
        # Read the full grid
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        # Read the text file and clean labels - one char per line
        labels = []
        lines = 0
        total_chars = 0
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    lines += 1
                    # Split line into individual characters
                    chars = list(line.strip())
                    total_chars += len(chars)
                    #print(f"Line {lines}: Found {len(chars)} characters: {chars}")
                    
                    labels.extend(chars)
                    #print(f"Added {len(chars)} characters")
        except UnicodeDecodeError:
            with open(txt_path, 'r', encoding='shift-jis') as f:
                for line in f:
                    chars = list(line.strip())
                    labels.extend([char for char in chars if char and char != '\x00'])
        labels = [jis_to_hiragana(char) for char in labels]
     #   print(f"Loaded {lines} lines")
    #    print(f"Total characters found: {total_chars}")
        print(f"Valid labels loaded: {len(labels)}")
        
        # Get image dimensions
        cell_height, cell_width = grid_size
        rows, cols = 40, 50
        
        expected_height = rows * cell_height
        expected_width = cols * cell_width
        if img.shape[:2] != (expected_height, expected_width):
            raise ValueError(f"Image dimensions must be {expected_width}x{expected_height}")
        
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        label_counts = {}
        
        # Iterate through the grid
        processed_count = 0
        for row in range(rows):
            for col in range(cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Crop the cell
                cell = img[y1:y2, x1:x2]
                
                # Get label for current cell
                if processed_count < len(labels):
                    label = labels[processed_count]
                    
                    if not label or label == '\x00':
                        class_dir = os.path.join(output_dir, '_null_')
                    else:
                        class_dir = os.path.join(output_dir, f"'{label}'")
                    
                    Path(class_dir).mkdir(parents=True, exist_ok=True)
                    
                    # Use '_null_' as filename prefix for null characters
                    # technically you can skip over if the label is null but it helps to keep
                    # these characters so you know if something went wrong
                    filename_prefix = '_null_' if not label or label == '\x00' else label
                    
                    # Initialize counter for new labels
                    # TODO - this will overwrite existing files with the same prefix

                    if filename_prefix not in label_counts:
                        label_counts[filename_prefix] = 0
                    
                    output_path = os.path.join(class_dir, f"'{filename_prefix}'_{label_counts[filename_prefix]:04d}.png")
                    if not cv2.imwrite(output_path, cell):
                        raise IOError(f"Failed to save image: {output_path}")
                    
                    label_counts[filename_prefix] += 1
                    processed_count += 1
                else:
                    break
        
        print(f"Successfully processed {processed_count} cells")
     #   print("Label distribution:", dict(label_counts))
                
    except Exception as e:
        print(f"Error in split_grid_image: {str(e)}")
        raise

def find_matching_pairs(base_dir):
    """Find all matching PNG and TXT pairs in the directory."""
    png_files = glob.glob(os.path.join(base_dir, "**/*.png"), recursive=True)
    pairs = []
    
    for png_file in png_files:
        txt_file = png_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(txt_file):
            pairs.append((png_file, txt_file))
    
    return pairs

def process_pair(args):
    """Process a single image/text pair."""
    img_path, txt_path, output_dir = args
    try:
        print(f"Processing {os.path.basename(img_path)}")
        split_grid_image(img_path, txt_path, output_dir)
        return f"Successfully processed {os.path.basename(img_path)}"
    except Exception as e:
        return f"Failed to process {os.path.basename(img_path)}: {str(e)}"

def main():
    base_dir = "/Users/chai/Downloads/ETL 仮名・漢字 dataset"
    output_dir = os.path.join(base_dir, "classes")
    
    # Find all matching pairs
    pairs = find_matching_pairs(base_dir)
    print(f"Found {pairs}")
    if not pairs:
        print("No matching PNG/TXT pairs found!")
        return
    
    print(f"Found {len(pairs)} pairs to process")
    
    process_args = [(png, txt, output_dir) for png, txt in pairs]
   
    with Pool(processes=8) as pool:
        results = pool.map(process_pair, process_args)
    

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
