# Method to pick the final expansions
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

# VQA
dataset = 'vqa'
images_path = 'data/vqa/train2014'
questions_path = 'data/vqa/v2_OpenEnded_mscoco_train2014_questions.json'
captions_path = 'data/vqa/expansion/captions_train_vqa.json'
captions_comet_expansions_path = 'data/vqa/expansion/caption_comet_expansions_train_vqa.json'
questions_comet_expansions_path = 'data/vqa/expansion/question_comet_expansions_vqa_175000.json'

save_sentences_caption_expansions = 'outputs/caption_expansion_sentences_train.json'
save_sentences_question_expansions = 'outputs/question_expansion_sentences_train.json'
method = 'SEMANTIC_SEARCH'  # [SEMANTIC_SEARCH, SEMANTIC_SEARCH_QN, SIMILARITY]

# for VCR these are the paths
# dataset = 'vcr'
# images_path = 'data/vcr/vcr1images'
# questions_path = 'data/vcr/vcr1annots/train.jsonl'
# captions_path = 'data/vcr/expansion/captions_train_vcr.json'
# captions_comet_expansions_path = 'data/vcr/expansion/caption_comet_expansions_train_vcr.json'
# questions_comet_expansions_path = 'data/vcr/expansion/question_comet_expansions_train_vcr.json'
#
# save_sentences_caption_expansions = 'outputs/caption_expansion_sentences_train_vcr.json'
# save_sentences_question_expansions = 'outputs/question_expansion_sentences_train_vcr.json'
# method = 'SEMANTIC_SEARCH'  # [SEMANTIC_SEARCH, SEMANTIC_SEARCH_QN, SIMILARITY]
#


def imageid_to_path(image_id, split='train'):
    n_zeros = 12 - len(image_id)
    filename = f'COCO_{split}2014_' + n_zeros*'0' + image_id + '.jpg'
    return filename


def image_path_to_id(image_fullname, split='train'):
    img_id = image_fullname.replace('COCO_train2014_000000', "")
    img_id = img_id.replace('.jpg', "")
    return str(int(img_id))