# mojinet 〜　文字・ネト
Deep learning Japanese character recognition model using [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) architecture, via transfer learning on the [ETL handwritten kanji/kana](http://etlcdb.db.aist.go.jp/?lang=ja) dataset. Includes simple　preprocessing utilities and training pipeline for Japanese OCR tasks


## Features
- ConvNeXt-based handwritten Japanese character recognition
- ETL文字 dataset preprocessing tools (Grid segmentation、labelling、formatting classes)
- Training/inference pipeline


# Working with the ETL cdb

### ```unpack.py```


built on top of the sample script provided to fix JIS mapping, add batched processing and QoL like logging. It outputs the image grid, a txt file containing the characters and a csv with the rest of the data in the binary. It populates each ETL folder with the extracted files. unpack.py supports multiprocessing with 8 default workers.

### ```python3 grid_to_single.py```
splices the grid(s) into single images. It creates and manages the following filesystem structure:

```
Dataset/
└── ETL_folders/

Root(where you execute from)/
└── output_dir/
    └── log.json
    └── ETL_folder#/
        └── character_label/
            ├── character_001.png
            ├── character_002.png
            └── character_003.png
```
where output_dir is the argument you pass to grid_to_single and log.json stores: 

→total processed/failed/labelled image info

→per-ETL folder (ETL1/2/3...) specifics 

### ```grid_walk.py```
I had some trouble with tag cleanup. Some are wrapped with \n characters, so only skipping 0x00A in your tag reading logic doesnt work.
grid_walk lets you examine specific grids on a per-image basis; it loads image by image, along with its position, the hex/utf8 rendition of the label from the txt that *should* correspond to it and context about the next & previous 5 chars. 
<img width="1612" alt="grid_walk" src="https://github.com/user-attachments/assets/b7ce83ec-9453-4ec1-ba09-e829464f14c3" />


## Notes
If you extract with the provided unpack all JIS (0201/0208) characters get converted to unicode and then further [normalised in the case of half or full width](https://www.unicode.org/charts/PDF/UFF00.pdf) depending on the file being processed. ETL1/6/7 all use half-width katakana for the labels; On ETL6, it maps to full-width katakana(as in, the character ア will respond to an ア in the image grid). However, on ETL7 it maps to hiragana (half-width ア in the text → あ in the image), so ETL6 only requires you normalise after converting to unicode, but ETL7 also needs to be offset to hiragana.
