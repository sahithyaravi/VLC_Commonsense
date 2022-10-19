# VLC-Commonsense
This repo is part of the VLC-BERT project (https://github.com/aditya10/VLC-BERT).


### Download data and organize expansions
Follow all the steps for data download and organization from https://github.com/aditya10/VLC-BERT/blob/master/QUICK_SETUP.md
mkdir data
cd data
ln -s DATA_PATH ./

### Install requirements
pip install -r requirements.txt

### Configure 
Set paths for images, COMET expansions, method to generate context in config.py

### To process expansions
python process_expansions.py

### To train S-BERT (Augmented S-BERT)
python train_sbert_search.py

