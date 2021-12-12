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
split = 'val2014'
images_path = f'data/vqa/{split}'
questions_path = f'data/vqa/questions/v2_OpenEnded_mscoco_{split}_questions.json'
captions_path = f'data/vqa/expansion/captions/captions_{split}_vqa.json'
captions_comet_expansions_path = f'data/vqa/expansion/caption_expansions/caption_comet_expansions_{split}_vqa.json'
questions_comet_expansions_path = f'data/vqa/expansion/question_expansions/question_comet_expansions_{split}_vqa.json'

save_sentences_caption_expansions = f'data/vqa/expansion/caption_expansion_sentences_{split}_{dataset}.json'
save_sentences_question_expansions = f'data/vqa/expansion/question_expansion_sentences_{split}_{dataset}.json'
method = 'SEMANTIC_SEARCH_QN'  # [1- SEMANTIC_SEARCH, 2-SEMANTIC_SEARCH_QN, 3-TOP]
final_expansion_save_path = f'outputs/picked_expansions_{method}_{dataset}_{split}.json'

save_top_caption_expansions = f'data/vqa/expansion/top_cap_sentences_{split}_{dataset}.json'
save_top_qn_expansions = f'data/vqa/expansion/top_qn_sentences_{split}_{dataset}.json'

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
# method = 'SEMANTIC_SEARCH'
#


def imageid_to_path(image_id):
    n_zeros = 12 - len(image_id)
    filename = f'COCO_{split}_' + n_zeros*'0' + image_id + '.jpg'
    return filename


def image_path_to_id(image_fullname):
    img_id = image_fullname.replace(f'COCO_{split}_00', "")
    img_id = img_id.replace('.jpg', "")
    return str(int(img_id))