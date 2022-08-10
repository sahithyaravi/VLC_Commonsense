# vlc_transformer
This repo is part of the VLC-BERT project.


### Download data and organize expansions
Follow all the steps for data download and organization from https://github.com/aditya10/VLC-BERT/blob/master/QUICK_SETUP.md
mkdir data
cd data
ln -s DATA_PATH ./

### Install requirements
pip install -r requirements.txt
Download coco model for captioning: gdown --id 1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX
(or link: https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view)
### Configure 
Set paths for images, expansions, method to generate context in config.py

### To process expansions
python process_expansions.py
