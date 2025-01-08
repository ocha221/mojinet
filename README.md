# Mojinet - 文字・ネト 文字認識 

Tooling & a deep learning model for Japanese character recognition using the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) architecture. Built through transfer learning on the [ETL文字データベース](http://etlcdb.db.aist.go.jp/?lang=ja).
## 概要 Overview 

The project is meant to streamline the process of working with the ETL Character Database (ETL-CDB), offering high-performance preprocessing tools and dataset preparation utilities for Japanese character recognition models. Near every task is parallelised with the MapReduce paradigm in mind.

## ✨ Key Features


- Complete ETL dataset preprocessing pipeline with parallel processing:
  - Unpacking
  - Tiling & dataset browsing
  - Merging
  - Dataset prep: Image Normalisation, Upscaling, splitting
- Robust character encoding handling (JIS → Unicode conversion & Normalisation)
- Visual debugging tools for dataset inspection
- Logging


##  Components

### Dataset Processing Pipeline

####  unpack.py: ETL Binary Extraction
- Parallel processing support via multiprocessing
- Intelligent JIS to Unicode character conversion
- Outputs:
  - Character image grids
  - UTF-8 character mapping files
  - Metadata CSVs with character information
- Comprehensive logging system

####  tiles_from_pairs.py: Grid Segmentation
- Segments character grids into individual samples
- Parallel grid processing with worker management
- Detailed success/failure statistics
- Logging
- Can process either the whole collection, or single ETL categories
```
Dataset/
└── ETL_folders/

output_dir/
├── worker_0/
│   └── grid_*/
│        └── symbol*/
│           ├── symbol*_00000.png
│           └── symbol*_00001.png
├── worker_1/
└── processing_log_YYYYMMDD_HHMMSS.json
```

####  merger.py: Dataset Consolidation
- Highly parallel, fast merging of worker outputs using os.sendfile
- Efficient label indexing system & directory scanning
- Character → grid mapping with comprehensive metadata
- Progress tracking with detailed statistics
- Dry-run capability for validation
```
final_dir/
├── character_1/           
│   ├── 0.png             
│   ├── 1.png
│   └── ...
├── character_2/
└── worker_*_stats.json
```
####  dataset_splitter.py: Dataset Preparation
- Interactive CLI for customizing dataset creation
- Configurable dataset sampling with percentage control
- Train/Val/Test splitting (80/10/10 default)
- Image processing:
  - Multiple upscaling methods (Lanczos/Bicubic/Bilinear)
  - Configurable target sizes (224x224, 384x384)
  - Optional image normalization
- Character normalization (full-width → standard)
- Fast!

#### 🔍 grid_walk.py: Visual Debugging

- Character-by-character grid examination
- Context visualization (previous/next 5 characters)
- Label verification
- Hex/UTF-8 label inspection
<img width="1612" alt="grid_walk" src="https://github.com/user-attachments/assets/b7ce83ec-9453-4ec1-ba09-e829464f14c3" />

##  Getting Started

### Prerequisites
tba


## Notes

If you extract with the provided unpack all JIS (0201/0208) characters get converted to unicode and then further [normalised in the case of half or full width](https://www.unicode.org/charts/PDF/UFF00.pdf) depending on the file being processed. ETL1/6/7 all use half-width katakana for the labels; On ETL6, it maps to full-width katakana(as in, the character ア will respond to an ア in the image grid). However, on ETL7 it maps to hiragana (half-width ア in the text → あ in the image), so ETL6 only requires you normalise after converting to unicode, but ETL7 also needs to be offset to hiragana.
