
import logging
# Configure logging here
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)


# Configure parameters for semantic search
# Methods to pick the final expansions
"""
@param method:
Sem_V1:
Get expansion for captions of each image [c1.....cn]
For each question of an image pick c1...ck by using semantic search
For each image pick c1....cm using image search


Sem_V2
Get expansion for captions [c1...cn] and question [q1...qn] and combine them [c1...cn q1...qn]
For each question perform semantic search to find e1...ek
intersect this with image?

Top
No semantic search, just pick topk expansions 

"""
method = 'sem2'  # [sem1- caption, sem2-caption+question]
version = '3'

"""
@param: model_for_qn_search
"clip" - uses clip for both image and semantic search
"text" - uses clip for image search and text model for semantic search
"""
model_for_qn_search = "text"


# Configure all paths here
dataset = 'okvqa'
split = 'train2014'
data_root = "/ubc/cs/research/nlp/sahiravi/datasets/"
images_path = f'{data_root}/vqa/{split}/'
questions_path = f'{data_root}/questions/OpenEnded_mscoco_{split}_questions.json'
question_csv = f'{data_root}/questions/OpenEnded_mscoco_{split}_questions1.csv'
captions_path = f'/ubc/cs/research/nlp/sahiravi/datasets/expansion/captions/captions_{split}_vqa.json'
captions_comet_expansions_path = f'/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/expansions_batch/caption_comet_expansions_{split}_vqa_v3.json'
questions_comet_expansions_path = f"/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/expansions_batch/okvqa_question_comet_expansions_{split}_vqa_v2.json"
caption_expansion_sentences_path = f'caption_expansion_sentences_{split}_vqa.json'
question_expansion_sentences_path = f'question_expansion_sentences_{split}_vqa.json'
final_expansion_save_path = f'{dataset}/{method}.{version}_{dataset}_{split}'
topk_caption_path = f'{dataset}/top_cap_sentences_{split}_{dataset}.json'
topk_qn_path = f'{dataset}/top_qn_sentences_{split}_{dataset}.json'
