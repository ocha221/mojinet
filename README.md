# Mojinet - 文字・ネト 文字認識 
### ETLCDB Extraction | Japanese OCR | Kanji Recognition 
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)


Tooling for the etl character database [ETL文字データベース](http://etlcdb.db.aist.go.jp/?lang=ja).
## 概要 Overview 

The project is meant to streamline the process of working with the ETL Character Database, offering simple, high-performance preprocessing tools and dataset preparation utilities for Japanese character recognition models. Near every task is parallelised with the MapReduce paradigm in mind.


## ✨ Key Features


- Fully featured ETLCDB extractor pipeline with parallel processing:
  - ETL binary unpacking, extraction
  - Grid tiling & dataset exploration
  - Multi-worker dataset merging
  - Preprocessing: Image Binarisation, Upscaling, Dataset Splitting & basic filtering
- Dataset validation tools
- logging

##  Getting Started

### Prerequisites
- Python 3.13
- [```JIS 0201```](https://www.unicode.org/anon-ftp/Public/MAPPINGS/OBSOLETE/EASTASIA/JIS/JIS0201.TXT) & [```JIS 0208```](https://www.unicode.org/anon-ftp/Public/MAPPINGS/OBSOLETE/EASTASIA/JIS/JIS0208.TXT) mappings (available from Unicode directly)
- Request access to the ETLCDB on the [site](http://etlcdb.db.aist.go.jp/?lang=ja)
- Download ```euc_co59.dat``` from the same site
- optionally edit manager.py to include the download urls from the AIST website to automate download/unzip. (recommended)
```
git clone https://github.com/ocha221/mojinet.git
cd mojinet/
pip3 install -r requirements.txt
python3 manager.py --help # display all commands
python3 manager.py pipeline # First run will take a second or two to initiate everything. Then it will run through the entire processing pipeline
```

<img src="https://github.com/user-attachments/assets/12b61d38-550a-4958-b329-2ac25c75186c" width="45%"></img> <img src="https://github.com/user-attachments/assets/98decf4e-f42b-4861-bb8c-c932a381e0a2" width="45%"></img> <img src="https://github.com/user-attachments/assets/93ac2dcc-4d71-43ce-93de-d7bdb8267f15" width="45%"></img> <img src="https://github.com/user-attachments/assets/a1e611db-f7cd-4b6c-b57d-7cde46097b28" width="45%"> </img><img width="1712" alt="" src="https://github.com/user-attachments/assets/1f4c55ea-f513-4ad8-9c28-09ff3e12eafc" />

##  Components

### Dataset Processing Pipeline

#### ```unpacking → grid segmentation → consolidation → preparation```

## manager.py: Unified runner

A helper utility that bridges the subscripts together. You can of course use each script in tools/ independently, but with ```manager.py pipeline``` the aim is to get from the packed binaries to a finalised dataset in ~300 seconds on my M1 macbook with 8 workers, with QoL like handling directory creation and chaining for you, and optionally automating the download of the etl files. For this reason, when running in ```pipeline``` mode, the arguments for every function except ```split``` are streamlined to use as many workers as possible and to extract to hardcoded directories; you'll still be prompted for the final dataset params. Pass the --cjk-only flag to ```pipeline``` to filter out latin characters from your final dataset.


####  unpack.py: ETL Binary Extraction
- Parallel processing support via multiprocessing
- Complete JIS X 0208/0201 to Unicode conversion
- Generates:
  - ETL character image grids
  - Unicode character mappings
  - ETL metadata in CSV format
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
  - Optional image binarisation (Otsu's method and inversion)
- Character normalization (full-width → standard)
- Non-CJK filtering
- Fast!

#### 🔍 grid_walk.py: Visual Debugging

- Interactive matplotlib character-by-character grid examination
- Context visualization (previous/next 5 characters)
- Label verification
- Hex/UTF-8 label inspection



All JIS (0201/0208) characters get converted to unicode and then further [normalised in the case of half or full width](https://www.unicode.org/charts/PDF/UFF00.pdf) depending on the file being processed. 
- ETL6: Unicode normalization only
- ETL7: Unicode normalization + hiragana offset

This is because ETL1/6/7 all use half-width katakana for the labels; On ETL6, it maps to full-width katakana(as in, the character ア will respond to an ア in the image grid). However, on ETL7 it maps to hiragana (half-width ア in the text → あ in the image), so ETL6 only requires you normalise after converting to unicode, but ETL7 also needs to be offset to hiragana.

## Technical Implementation Details


The ETL database consists of 11 distinct folders, each containing multiple binary files that store packed image data, labels, and associated metadata. 

The initial unpacking process generates image grids alongside their corresponding labels txt files (1091 pairs). I decided to do it this way because it helped with debugging, plus this way you can look through the dataset(grid_walk!), though it will of course take up a little more space on your drive. 

The resulting grids then get tiled into their respective characters. Since theres a lot (1.9 ish million), it helps to parallelise. The issue lies in that grid 1 and grid 99 could both have a label for “あ”, so we'd need a lock and a way to avoid overwritting, which is a massive slowdown. Instead, the workers all get their own folder, and create a new subfolder for each pair the process. this subfolder contains all the labels. So multiple grids get processed in parallel and safely unpacked in their own folder.

However, this leaves us with a mess. We have 1091 grid directories of labels spread across N worker folders;

The merger solves this problem very efficiently. It starts with a single pass over all the worker/grid_folder/labels structures and creates a dictionary of (label =grid_folder(s)):
```
{
    "あ”："paths": "<worker1/grid_8>","<worker4/ETL9B_01_01>", ...,
    ...
}
```
This takes about 5 seconds on my ssd. Now we know where each label is, and since we have this map we can safely assign each LABEL(or chunk of labels) to a worker process, which then merges & combines the different directories into a single folder with an incrementing counter. So theres no synchronising required, no locking and no shared counters, each worker is guaranteed to never overlap with someone else.

I originally used shutil.copy2() but os.sendfile() is dramatically faster. On my ssd this took about 150 seconds to finish.

dataset_splitter takes the final dataset from the merge and structures it. The cli is bascially self explanatory. You can choose which % of the dataset you want to extract, if you'd like to upscale (i used the convnext dimensions as options) and with which method, and optionally normalise ( / 255. ). You can also filter out non-cjk characters.

depending on your linter/pylance strictness, ```f.bytepos = pos * self.octets_per_record``` in ```unpack.py``` might warn/error out, but the program works as normal.

note: Commit ```517217b``` additionally fixed the stats reporting issue with the merger. now it will show correct processing counts.
