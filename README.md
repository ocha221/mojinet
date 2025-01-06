# mojinet 〜　文字・ネト
Deep learning Japanese character recognition model using [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) architecture, via transfer learning on the [ETL handwritten kanji/kana](http://etlcdb.db.aist.go.jp/?lang=ja) dataset. Includes simple　preprocessing utilities and training pipeline for Japanese OCR tasks


## Features
- ConvNeXt-based handwritten Japanese character recognition
- ETL文字 dataset preprocessing tools (Grid segmentation、labelling、formatting classes)
- Training/inference pipeline


# Working with the ETL cdb

## Key features

→Fully parallelised using processes to avoid GIL (FS i/o is a smaller bottleneck than GIL) 
→Character grid extraction and tiling
→Support for all ETL dataset types (ETL1-9)
→Fixed JIS to Unicode character conversion
→Visual debugging tools for grid inspection

✅Binary ETL file unpacking (unpack.py)　& character mapping 
✅Image grid extraction (tiles_from_pairs.py)
✅Grid debugging  (grid_walk.py)
✅Multi-worker output merging (merger.py)


### ```unpack.py```

built on top of the sample script provided to fix JIS mapping, add batched processing and QoL like logging. It outputs the image grid, a txt file containing the characters and a csv with the rest of the data in the binary. It populates each ETL folder with the extracted files. unpack.py supports multiprocessing.

### ```tiles_from_pairs.py```
Slices the grid images into individual character tiles with proper labeling. Supports all ETL dataset image sizes and can process specific ETL types or the entire collection. Can process multiple grids in parallel or a single file. Generates detailed logs including success/failure statistics. It manages the following structure:

```
Dataset/
└── ETL_folders/

output_dir/
├── worker_0/
│   └── grid_*/
│        └──symbol*/
│           ├──symbol*_00000.png
│           └──symbol*_00001.png
├── worker_1/
│    ...
└── processing_log_YYYYMMDD_HHMMSS.json
```

where output_dir is the directory you specify (default temp_workers and processing_log.json stores: 

→total processed/failed/labelled image info

→detailed per-grid info. 

### ```merger.py```

Merges the scattered labels into a unified indexed structure. Generates an index of labels out of tiles_from_pairs' output so that merging can be parallelised. Dry-run capability for operation verification before commiting to disk. Very efficient search (only takes about 5 seconds to map everything), but the final execution time will depend on your ssd speed. On my M1 macbook, it takes about 300 seconds to write the whole dataset when using 8 workers.

```
Input Structure:
output_dir/
└── worker_*/                
    └── grid_*/             
        └── character_*/    
            ├── image1.png
            └── image2.png

Output Structure:
final_dir/
├── character_1/           
│   ├── 0.png             
│   ├── 1.png
│   └── ...
├── character_2/
│   └── ...
└── worker_*_stats.json  
```
### ```grid_walk.py```
I had some trouble with tag cleanup. Some are wrapped with \n characters, so only skipping 0x00A in your tag reading logic doesnt work.
grid_walk lets you examine specific grids on a per-image basis; it loads image by image, along with its position, the hex/utf8 rendition of the label from the txt that *should* correspond to it and context about the next & previous 5 chars. 
<img width="1612" alt="grid_walk" src="https://github.com/user-attachments/assets/b7ce83ec-9453-4ec1-ba09-e829464f14c3" />


## Notes
If you extract with the provided unpack all JIS (0201/0208) characters get converted to unicode and then further [normalised in the case of half or full width](https://www.unicode.org/charts/PDF/UFF00.pdf) depending on the file being processed. ETL1/6/7 all use half-width katakana for the labels; On ETL6, it maps to full-width katakana(as in, the character ア will respond to an ア in the image grid). However, on ETL7 it maps to hiragana (half-width ア in the text → あ in the image), so ETL6 only requires you normalise after converting to unicode, but ETL7 also needs to be offset to hiragana.
