# Research in DS: Graph Neural Networks for Scene Understanding

## Setting up the environment

```
conda create --name ds --file requirements.txt
conda activate ds
conda install pyg -c pyg
```

## Downloading the pre-processed Hypersim dataset

Download the processed dataset from https://polybox.ethz.ch/index.php/s/UEWtvTF67ycntGP and extract to main folder.
Note that this is the result from the process method in dataset.py.


## Training the model
Edit model configurations in config.yaml.
To train the model:
```
cd ml-hypersim
python main.py --mode train
```