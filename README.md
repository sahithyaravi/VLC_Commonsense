# vlc_transformer
This repo has the code for generating captions,
processing expansions.
COMET expansion generation is under a separate repo.

### Download data
- Save/link vcr images and questions under data/vqa, data/vcr
- Save/link  the expansions, captions under data/vqa/expansion, data/vcr/expansion

### Install requirements
pip install -r requirements.txt

### Configure 
Set paths for images, expansions, method to generate context in config.py

### To process expansions
Run process_expansions.py
The picked expansions will be saved under outputs/


