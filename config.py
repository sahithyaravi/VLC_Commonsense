import logging

# Configure logging here
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure parameters for semantic search
# Methods to pick the final expansions
method = 'sem-q'  # [sem1- caption, sem2-caption+question]
version = '1'  # version of semantic search results
dataset = 'aokvqa'  # dataset 'vcr', 'okvqa' or 'aokvqa
data_root = "scratch/data"  # root of the dataset folder arranged similar to VLC-BERT
split = 'test'

"""
@param: model_for_qn_search
"clip" - uses clip for both image and semantic search
"text" - uses clip for image search and text model for semantic search
"""

# Configure all paths here
if dataset == "okvqa":
    images_path = f'{data_root}/coco/{split}2014/'
    questions_path = f'{data_root}/coco/okvqa/OpenEnded_mscoco_{split}_questions.json'
    question_csv = questions_path.split("json")[0] + 'csv'
    captions_path = f'{data_root}/coco/okvqa/commonsense/captions/captions_{split}_vqa.json'
    captions_comet_expansions_path = f'{data_root}/coco/okvqa/commonsense/expansions/caption_comet_expansions_{split}_vqa_v3.json'
    questions_comet_expansions_path = f'{data_root}/coco/okvqa/commonsense/expansions/okvqa_question_comet_expansions_{split}_vqa_v3.json'
    caption_expansion_sentences_path = f'{data_root}/coco/okvqa/commonsense/expansions/caption_expansion_sentences_{split}_vqa.json'
    question_expansion_sentences_path = f'{data_root}/coco/okvqa/commonsense/expansions/question_expansion_sentences_{split}_vqa.json'
    final_expansion_save_path = f'{data_root}/coco/okvqa/commonsense/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/coco/okvqa/commonsense/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/coco/okvqa/commonsense/expansions/top_qn_sentences_{split}_{dataset}.json'
elif dataset == 'vcr':
    images_path = f"{data_root}/vcr/vcr1images/"
    questions_path = f'{data_root}/vcr/{split}.jsonl'
    question_csv = questions_path.split("jsonl")[0] + 'csv'
    captions_path = f'{data_root}/vcr/commonsense/captions/captions_{split}_{dataset}.json'
    captions_comet_expansions_path = f'{data_root}/vcr/commonsense/expansions/caption_comet_expansions_{split}_{dataset}.json'
    questions_comet_expansions_path = f'{data_root}/vcr/commonsense/expansions/question_comet_expansions_{split}_{dataset}.json'
    caption_expansion_sentences_path = f'{data_root}/vcr/commonsense/expansions/caption_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    question_expansion_sentences_path = f'{data_root}/vcr/commonsense/expansions/question_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    final_expansion_save_path = f'{data_root}/vcr/commonsense/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/vcr/commonsense/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/vcr/commonsense/expansions/top_qn_sentences_{split}_{dataset}.json'
elif dataset == 'aokvqa':
    images_path = f'{data_root}/coco/{split}2017/'
    questions_path = f'{data_root}/coco/aokvqa/aokvqa_v1p0_{split}.json'
    question_csv = questions_path.split("json")[0] + 'csv'
    captions_path = f'{data_root}/coco/aokvqa/commonsense/captions/captions_{split}_{dataset}.json'
    captions_comet_expansions_path = f'{data_root}/coco/aokvqa/commonsense/expansions/caption_comet_expansions_{split}_{dataset}.json'
    questions_comet_expansions_path = f'{data_root}/coco/aokvqa/commonsense/expansions/question_comet_expansions_{split}_{dataset}.json'
    caption_expansion_sentences_path = f'{data_root}/coco/aokvqa/commonsense/expansions/caption_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    question_expansion_sentences_path = f'{data_root}/coco/aokvqa/commonsense/expansions/question_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    final_expansion_save_path = f'{data_root}/coco/aokvqa/commonsense/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/coco/aokvqa/commonsense/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/coco/aokvqa/commonsense/expansions/top_qn_sentences_{split}_{dataset}.json'
else:
    raise ValueError("Dataset must be one of okvqa, vcr or aokvqa")
