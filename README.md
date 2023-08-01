# Research in DS: Graph Neural Networks for Scene Understanding

## 1. Setting up the environment

```
conda create --name ds --file requirements.txt
conda activate ds
conda install pyg -c pyg
```

## 2. Downloading the pre-processed Hypersim dataset

Download the processed dataset from https://polybox.ethz.ch/index.php/s/UEWtvTF67ycntGP and extract to main folder.
Note that this is the result from the process method in dataset.py. 
To view each graph's corresponding image, download all relevant images from the raw dataset:
```
cd contrib/9991/
python download.py --contains tonemap.jpg --silent
```


## 3. Training the model
Edit model configurations in config.yaml.
To train the model:
```
cd ml-hypersim
python main.py --mode train
```

## 4. debugging