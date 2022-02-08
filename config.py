import logging
# Configure logging here
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)


# Configure Methods for semantic search here
# Methods to pick the final expansions
"""
SEMANTIC_SEARCH:
Get expansion for captions of each image [c1.....cn]
For each question of an image pick c1...ck by using semantic search (https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)

SIMILARITY
Get expansion for captions [c1...cn] and question [q1...qn]
Find most similar entries and return c1...ck or q1...qk

SEMANTIC_SEARCH_QN
Get expansion for captions [c1...cn] and question [q1...qn] and combine them [c1...cn q1...qn]
For each question perform semantic search to find e1...ek

"""
method = 'SEMANTIC_SEARCH'  # [1- SEMANTIC_SEARCH, 2-SEMANTIC_SEARCH_QN]

# Configure all paths here
dataset = 'vqa'
split = 'train2014'
data_root = "/Users/sahiravi/Documents/research/VL project/vqa_all_data/data/"
images_path = f'{data_root}/vqa/{split}'
questions_path = f'{data_root}/ok-vqa/OpenEnded_mscoco_{split}_questions.json'
captions_path = f'{data_root}/vqa/expansion/captions/captions_{split}_vqa.json'
captions_comet_expansions_path = f'{data_root}/vqa/expansion/caption_expansions/caption_comet_expansions_{split}_vqa_v2.json'
questions_comet_expansions_path = ""
save_sentences_caption_expansions = f'caption_expansion_sentences_{split}_{dataset}.json'
save_sentences_question_expansions = f'question_expansion_sentences_{split}_{dataset}.json'
final_expansion_save_path = f'temps/picked_expansions_{method}_{dataset}_{split}_V2.json'
save_top_caption_expansions = f'temps/top_cap_sentences_{split}_{dataset}.json'
save_top_qn_expansions = f'temps/top_qn_sentences_{split}_{dataset}.json'




