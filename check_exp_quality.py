import json
import string
from utils import imageid_to_path
from pytorch_pretrained_bert import BertTokenizer
from nltk.corpus import stopwords

# nltk.download('stopwords')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
s = set(stopwords.words('english'))


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_exp_quality_aok(exp_name, set_name):
    # Load questions:
    annotations = _load_json('scratch/data/coco/aokvqa/aokvqa_v1p0_' + set_name + '.json')

    # Load expansions:
    expansions = _load_json('scratch/data/coco/aokvqa/commonsense/expansions/' + exp_name + '_aokvqa_' + set_name + '.json')

    helpful = 0
    total = 0
    common_tokens = {}

    for q in annotations:
        q_id = q['question_id']
        im_id = q['image_id']

        ans_text = ' '.join(q['direct_answers'])
        ans_text = ans_text.translate(str.maketrans('', '', string.punctuation))
        ans_text = ans_text.lower()

        exp_text = expansions['{:012d}.jpg'.format(im_id)][str(q_id)][0]
        exp_text = exp_text.split('.')[:5]
        exp_text = ' '.join(exp_text)

        exp_text = exp_text.translate(str.maketrans('', '', string.punctuation))
        exp_text = exp_text.lower()

        ans_tokens = tokenizer.tokenize(ans_text)
        ans_tokens = [t for t in ans_tokens if t not in s]
        exp_tokens = tokenizer.tokenize(exp_text)
        exp_tokens = [t for t in exp_tokens if t not in s]

        is_helpful = False
        for token in ans_tokens:
            if token in exp_tokens:
                is_helpful = True
                if token not in common_tokens:
                    common_tokens[token] = 1
                else:
                    common_tokens[token] += 1

        if is_helpful:
            helpful += 1
        total += 1

    common_tokens = dict(sorted(common_tokens.items(), key=lambda item: item[1], reverse=True))
    common_tokens = list(common_tokens.keys())[:10]

    return helpful, total, common_tokens

def check_exp_quality_ok(exp_name, set_name):
    # Load questions:
    annotations = _load_json(f'scratch/data/coco/annotations/mscoco_{set_name}2014_annotations.json')["annotations"]

    # Load expansions:
    expansions = _load_json('scratch/data/coco/okvqa/commonsense/expansions/' + exp_name + '_okvqa_' + set_name + '.json')

    helpful = 0
    total = 0
    common_tokens = {}

    for q in annotations:
        q_id = q['question_id']
        image_id = str(q['image_id'])
        ans = q["answers"]
        ans_text = ' '.join([a['answer'] for a in ans])
        ans_text = ans_text.translate(str.maketrans('', '', string.punctuation))
        ans_text = ans_text.lower()
        n_zeros = 12 - len(str(image_id))
        filename = f'COCO_{set_name}2014_' + n_zeros * '0' + image_id + '.jpg'
        try:
            exp_text = expansions[filename][str(q_id)][0]
        except KeyError:
            exp_text = expansions[image_id][str(q_id)]
        exp_text = exp_text.split('.')[:5]
        exp_text = ' '.join(exp_text)

        exp_text = exp_text.translate(str.maketrans('', '', string.punctuation))
        exp_text = exp_text.lower()

        ans_tokens = tokenizer.tokenize(ans_text)
        ans_tokens = [t for t in ans_tokens if t not in s]
        exp_tokens = tokenizer.tokenize(exp_text)
        exp_tokens = [t for t in exp_tokens if t not in s]

        is_helpful = False
        for token in ans_tokens:
            if token in exp_tokens:
                is_helpful = True
                if token not in common_tokens:
                    common_tokens[token] = 1
                else:
                    common_tokens[token] += 1

        if is_helpful:
            helpful += 1
        total += 1

    common_tokens = dict(sorted(common_tokens.items(), key=lambda item: item[1], reverse=True))
    common_tokens = list(common_tokens.keys())[:10]

    return helpful, total, common_tokens

if __name__ == '__main__':

    exp_names = ['semq.2']
    sets = ['train', 'val']
    datasets = ["okvqa", "aokvqa"]
    for dataset in datasets:
        for exp_name in exp_names:
            for set_name in sets:
                if dataset == "okvqa":
                    helpful, total, common_tokens = check_exp_quality_ok(exp_name, set_name)
                else:
                    helpful, total, common_tokens = check_exp_quality_aok(exp_name, set_name)
                print('{} {} {}: {}/{} ({:.2f}%)'.format(dataset, exp_name, set_name, helpful, total, helpful / total * 100),
                      ' Most common tokens: ', common_tokens)
                print()
