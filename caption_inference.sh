# script to run on compute canada
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install transformers
pip install git+https://github.com/openai/CLIP.git
python3 clip_prefix_captioning_inference.py

