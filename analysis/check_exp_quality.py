import json
import string

from nltk.corpus import stopwords
from pytorch_pretrained_bert import BertTokenizer

# nltk.download('stopwords')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
s = set(stopwords.words('english'))

root = "/Users/sahiravi/Documents/Research/VL project/scratch/data/coco"


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_exp_quality_aok(exp_name, set_name):
    # Load questions:
    annotations = _load_json(f'{root}/aokvqa/aokvqa_v1p0_' + set_name + '.json')

    # Load expansions:
    expansions = _load_json(f'{root}/aokvqa/commonsense/expansions/' + exp_name + '_aokvqa_' + set_name + '.json')
    raw_expansions = _load_json(f'{root}/aokvqa/commonsense/expansions/question_expansion_sentences_{set_name}_aokvqa_{exp_name}.json')

    K = 10
    helpful = 0
    total = 0
    common_tokens = {}
    print("USING K", K)
    for q in annotations:
        q_id = q['question_id']
        im_id = q['image_id']

        ans_text = ' '.join(q['direct_answers'])
        ans_text = ans_text.translate(str.maketrans('', '', string.punctuation))
        ans_text = ans_text.lower()

        exp_text = expansions['{:012d}.jpg'.format(im_id)][str(q_id)][0]
        exp_text = exp_text.split('.')[:K]
        exp_text = ' '.join(exp_text)

        raw_exp_text = ' '.join(raw_expansions[str(q_id)])
        exp_text = raw_exp_text
        print(len(raw_expansions[str(q_id)]))

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
    common_tokens = list(common_tokens.keys())[:15]

    return helpful, total, common_tokens


def check_exp_quality_ok(exp_name, set_name):
    # Load questions:
    annotations = _load_json(f'{root}/annotations/mscoco_{set_name}2014_annotations.json')["annotations"]

    # Load expansions:
    expansions = _load_json(
        f'{root}/okvqa/commonsense/expansions/{exp_name}/' + exp_name + '_okvqa_' + set_name + '.json')

    # Load raw expansions:
    # raw_expansions = _load_json(f'{root}/okvqa/commonsense/expansions/question_expansion_sentences_{set_name}_okvqa_{exp_name}.json')
    helpful = 0
    total = 0
    common_tokens = {}
    raw_commons = {}
    K = 10
    print("USING K", K)
    for q in annotations:
        q_id = q['question_id']
        image_id = str(q['image_id'])
        ans = q["answers"]
        ans_text = ' '.join([a['answer'] for a in ans])
        ans_text = ans_text.translate(str.maketrans('', '', string.punctuation))
        ans_text = ans_text.lower()
        n_zeros = 12 - len(str(image_id))
        filename = f'COCO_{set_name}2014_' + n_zeros * '0' + image_id + '.jpg'
        # raw_exp_text = ' '.join(raw_expansions[str(q_id)])

        try:
            exp_text = expansions[filename][str(q_id)][0]
        except KeyError:
            exp_text = expansions[image_id][str(q_id)]
        exp_text = exp_text.split('.')[:K]
        picked_exp_text = ' '.join(exp_text)

        exp_text = picked_exp_text

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

    exp_names = ['semq.4']
    sets = ['val', 'train']
    datasets = ["aokvqa"]
    for dataset in datasets:
        for exp_name in exp_names:
            for set_name in sets:
                if dataset == "okvqa":
                    helpful, total, common_tokens = check_exp_quality_ok(exp_name, set_name)
                else:
                    helpful, total, common_tokens = check_exp_quality_aok(exp_name, set_name)
                print('{} {} {}: {}/{} ({:.2f}%)'.format(dataset, exp_name, set_name, helpful, total,
                                                         helpful / total * 100),
                      ' Most common tokens: ', common_tokens)
                print()
