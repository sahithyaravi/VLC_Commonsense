
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
method = 'sem1'  # [sem1- caption, sem2-caption+question]
version = '3'  # version of semantic search results
dataset = 'okvqa'

"""
@param: model_for_qn_search
"clip" - uses clip for both image and semantic search
"text" - uses clip for image search and text model for semantic search
"""
model_for_qn_search = "text"
data_root = "data/"
# Configure all paths here
if dataset == "okvqa":
    split = 'train2014'
    images_path = f'{data_root}/coco/{split}/'
    questions_path = f'{data_root}/okvqa/OpenEnded_mscoco_{split}_questions.json'
    question_csv = f'{data_root}/okvqa/OpenEnded_mscoco_{split}_questions1.csv'
    captions_path = f'{data_root}/okvqa/captions/captions_{split}_vqa.json'
    captions_comet_expansions_path = f'{data_root}/okvqa/expansions/caption_comet_expansions_{split}_vqa_v3.json'
    questions_comet_expansions_path = f'{data_root}/okvqa/expansions/okvqa_question_comet_expansions_{split}_vqa_v3.json'
    caption_expansion_sentences_path = f'{data_root}/okvqa/expansions/caption_expansion_sentences_{split}_vqa.json'
    question_expansion_sentences_path = f'{data_root}/okvqa/expansions/question_expansion_sentences_{split}_vqa.json'
    final_expansion_save_path = f'{data_root}/okvqa/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/okvqa/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/okvqa/expansions/top_qn_sentences_{split}_{dataset}.json'
elif dataset == 'vcr':
    split = 'val'
    images_path = f"{data_root}/vcr/vcr1images/"
    questions_path = f'{data_root}/vcr/{split}.jsonl'
    question_csv =  f'{data_root}/vcr/{split}.csv'
    captions_path =  f'{data_root}/vcr/captions_{split}_{dataset}.json'
    captions_comet_expansions_path =   f'{data_root}/vcr/expansions/caption_comet_expansions_{split}_{dataset}.json'
    questions_comet_expansions_path =  f'{data_root}/vcr/expansions/question_comet_expansions_{split}_{dataset}.json'
    caption_expansion_sentences_path =  f'{data_root}/vcr/expansions/caption_expansion_sentences_{split}_{dataset}.json'
    question_expansion_sentences_path = f'{data_root}/vcr/expansions/question_expansion_sentences_{split}_{dataset}.json'
    final_expansion_save_path = f'{data_root}/vcr/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path =  f'{data_root}/vcr/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path =  f'{data_root}/vcr/expansions/top_qn_sentences_{split}_{dataset}.json'
elif dataset == 'aokvqa':
    split = 'train'
    images_path = f'{data_root}/coco/{split}2017/'
    questions_path = f'{data_root}/aokvqa/aokvqa_v1p0_{split}.json'
    question_csv = f'{data_root}/aokvqa/{dataset}_{split}_questions.csv'
    captions_path = f'{data_root}/aokvqa/captions_{split}_{dataset}.json'
    captions_comet_expansions_path =f'{data_root}/aokvqa/expansions/caption_comet_expansions_{split}_{dataset}.json'
    questions_comet_expansions_path = f""
    caption_expansion_sentences_path = f'{data_root}/aokvqa/expansions/caption_expansion_sentences_{split}_{dataset}.json'
    question_expansion_sentences_path = f''
    final_expansion_save_path = f'{data_root}/aokvqa/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/aokvqa/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/aokvqa/expansions/top_qn_sentences_{split}_{dataset}.json'
elif dataset == 'fvqa':
    split = 'all'
    data_root = "/ubc/cs/research/nlp/sahiravi/datasets"
    images_path = f'{data_root}/fvqa/new_dataset_release/images/'
    questions_path = f'{data_root}/questions/all_qs_dict_release.json'
    question_csv = f'{data_root}/questions/fvqa_questions_{split}.csv'
    captions_path = f'/ubc/cs/research/nlp/sahiravi/datasets/expansion/captions/captions_{split}_fvqa.json'
    captions_comet_expansions_path = f'/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/expansions_batch/caption_comet_expansions_{split}_fvqa.json'
    questions_comet_expansions_path = f""
    caption_expansion_sentences_path = f'caption_expansion_sentences_{split}_{dataset}.json'
    question_expansion_sentences_path = f'question_expansion_sentences_{split}_{dataset}.json'
    final_expansion_save_path = f'{dataset}/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{dataset}/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{dataset}/top_qn_sentences_{split}_{dataset}.json'
else:
    raise ValueError("Dataset must be one of okvqa, vcr or aokvqa")