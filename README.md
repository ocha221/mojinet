# mojinet 〜　文字・ネト
Deep learning Japanese character recognition model using [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) architecture, via transfer learning on the [ETL handwritten kanji/kana](http://etlcdb.db.aist.go.jp/?lang=ja) dataset. Includes simple　preprocessing utilities and training pipeline for Japanese OCR tasks


## Features
- ConvNeXt-based handwritten Japanese character recognition
- ETL文字 dataset preprocessing tools (Grid segmentation、labelling、formatting classes)
- Training/inference pipeline

```python3 unpack.py <your etl Directory> or --single <path to a single ETL binary> ```
built on top of the sample script provided to fix JIS mapping, add batched processing and QoL like logging. It outputs the image grid, a txt file containing the characters and a csv with the rest of the data in the binary. 

```python3 grid_to_singly.py```
splices the grid(s) into single images. It creates and manages the following filesystem structure:

|<ETL folders>
|Classes 
-|<character label>
--|<character>_<incrementing count>.png




