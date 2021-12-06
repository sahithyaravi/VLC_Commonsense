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

### To do:
You can find the "done" items in the drive link
VQA:
#### Training set:
- Captions - done 
- Caption-expansions - done 
- Question -expansions - only until 175000
- Pick expansions using strategy 1 - done
- Pick expansions using strategy 3 - in progress
#### Val set:
- Captions - done
- Caption - expansions - in progress
- Question expansions - no

#### Test set:
- Captions - No
- Caption - expansions - no
- Question-expansions - no

VCR:
#### Training set:
- Captions - done 
- Caption-expansions - done 
- Question -expansions - no
- Pick expansions using strategy 1 - no
- Pick expansions using strategy 3 - no
#### Val set:
- Captions - done
- Caption - expansions - no
- Question expansions - no

#### Test set:
- Captions - No
- Caption - expansions - no
- Question-expansions - no