import json
import os
import random
import string
import nltk
from nltk.corpus import stopwords
from pytorch_pretrained_bert import BertTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

nltk.download('stopwords')
root = "/Users/sahiravi/Documents/Research/VL project/scratch/data/coco"
root = "/ubc/cs/research/nlp/sahiravi/vlc_transformer/scratch/data/coco"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
s = set(stopwords.words('english'))
semantic_search_model = "msmarco-roberta-base-ance-firstp"

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _intersection(lst1, lst2):
    return len(set(lst1).intersection(lst2)) > 0


def prepare_data(exp_name, set_name, base_dir):
    # Load questions:
    annotations = _load_json(
        f'{root}/aokvqa/aokvqa_v1p0_' + set_name + '.json')

    # Load expansions:
    expansions = _load_json(
        f'{root}/aokvqa/commonsense/expansions/' + exp_name + '_aokvqa_' + set_name + '.json')

    raw_expansions = _load_json(
        f'{root}/aokvqa/commonsense/expansions/question_expansion_sentences_{set_name}_aokvqa_{exp_name}.json')

    good_data_list = []
    bad_data_list = []
    data_list = []

    for q in annotations:
        q_id = q['question_id']
        im_id = q['image_id']
        raw_ques = q['question']

        ans_text = ' '.join(q['direct_answers'])
        ans_text = ans_text.translate(str.maketrans('', '', string.punctuation))
        ans_text = ans_text.lower()

        ans_tokens = tokenizer.tokenize(ans_text)
        ans_tokens = [t for t in ans_tokens if t not in s]

        #exps = expansions['{:012d}.jpg'.format(im_id)][str(q_id)][0]
        #exps = exps.split('.')
        
        exps = raw_expansions[str(q_id)]
        
        exps = [e.strip() + '.' for e in exps]

        for exp_text in exps:

            raw_text = exp_text

            exp_text = exp_text.translate(str.maketrans('', '', string.punctuation))
            exp_text = exp_text.lower()

            exp_tokens = tokenizer.tokenize(exp_text)
            exp_tokens = [t for t in exp_tokens if t not in s]

            if _intersection(ans_tokens, exp_tokens):
                good_data_list.append({'sents': [raw_ques, raw_text], 'label': 0.8})
            else:
                bad_data_list.append({'sents': [raw_ques, raw_text], 'label': 0.2})

    random.shuffle(bad_data_list)
    bad_data_list = bad_data_list[:len(good_data_list)]
    data_list = good_data_list + bad_data_list
    random.shuffle(data_list)

    print('good:', len(good_data_list))
    print('bad:', len(bad_data_list))
    print('total:', len(data_list))

    # Save data:
    savepath = os.path.join(base_dir, exp_name + '_aokvqa_' + set_name + '.json')
    with open(savepath, 'w') as f:
        json.dump(data_list, f, indent=4)

    return savepath


def train(train_file, base_dir):
    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(semantic_search_model)

    # Define your train examples. You need more than just two examples...
    train_egs = _load_json(train_file)
    train_examples = [InputExample(texts=e['sents'], label=e['label']) for e in train_egs]

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100, output_path=base_dir)


def test(base_dir):
    model = SentenceTransformer(base_dir)

    # Two lists of sentences
    sentences1 = ['What is the mood of the cow?',
                  'What is on the table?',
                  'What is missing from the dinner?']

    sentences2 = ['the cow is happy',
                  'a woman watches TV',
                  'man is capable of eating dinner']

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    for i in range(len(sentences1)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))


if __name__ == '__main__':

    base_path = 'sbert/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    savepath = prepare_data('semq.2', 'train', base_path)
    train(savepath, base_path)
    test(base_path)
