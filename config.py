images_path = 'data/train2014'
captions_path = 'data/captions_train.json'
captions_comet_expansions_path = 'data/caption_comet_expansions_train.json'

questions_path = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
questions_comet_expansions_path = 'data/question_comet_expansions_175000.json'

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
method = 'SEMANTIC_SEARCH_QN'  # [SEMANTIC_SEARCH, SEMANTIC_SEARCH_QN, SIMILARITY]

path_caption_expansions = 'caption_expansion_sentences_train.json'
path_question_expansions = 'question_expansion_sentences_train.json'
