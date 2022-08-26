import json
import os
import random
import string

from nltk.corpus import stopwords
from pytorch_pretrained_bert import BertTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# nltk.download('stopwords')



def refine_triples():
    """
    Returns
    -------
    """
    triples = pd.read_csv("svo_triples.csv")
    triples.drop_duplicates(inplace=True, ignore_index=True, subset=['subject', 'object'])
    triples.dropna(inplace=True, subset=['object'])

    indices = []
    # filtering
    for idx, row in triples.iterrows():
        subj = row["subject"].lower()
        obj = row["object"].lower()
        verb = row["relation"].lower()
        if subj in stop_words or subj.isdigit() or "$" in subj:
            indices.append(idx)
        if len(obj) < 3 or len(subj) < 3:
            indices.append(idx)
        if obj in stop_words:
            indices.append(idx)
        if "answer" in obj or "seen" in obj:
            indices.append(idx)

    triples.drop(triples.index[indices], inplace=True, axis=0)

    # Remove duplicates
    triples["length_o"] = triples["object"].str.len()
    triples = triples.sort_values("length_o", ascending=False).drop_duplicates(subset=["subject", "relation"],
                                                                               keep="first")
    triples["length_s"] = triples["subject"].str.len()
    triples = triples.sort_values("length_s", ascending=False).drop_duplicates(subset=["object"],
                                                                               keep="first")
    triples.reset_index(inplace=True)
    triples.to_csv("svo_triples_filtered.csv")

    # Map to relations
    indices_2 = []
    verbs_relations = [
        ("can", "CapableOf"),
        ("could", "CapableOf"),
        ("has", "HasProperty"),
        ("have", "HasProperty"),
        ("is in", "HasProperty"),
        ("always", "HasProperty"),
        ("made", "MadeOf"),
        ("consists", "MadeOf"),
        ("part", "PartOf"),
        ("so", "Causes"),
        ("use", "UsedFor"),
        ("uses", "UsedFor"),
        ("used", "UsedFor"),
        ("located", "AtLocation"),
        ("is on", "AtLocation"),
        ("is at", "AtLocation"),
        ("behind", "LocatedNear"),
        ("near", "LocatedNear"),
        ("is with", "LocatedNear"),
        ("next to", "LocatedNear"),
        ("not", "NotHasProperty"),
        ("don't", "NotHasProperty"),
        ("want", "XWant"),
        ("was", "isBefore"),
        ("were", "isBefore"),
        ("will", "isAfter"),
        ("is for", "ObjectUse"),
        ("because", "XReason")
    ]

    for idx, row in triples.iterrows():
        subj = row["subject"]
        obj = row["object"].lower()
        v = row["relation"].lower()
        doc = nlp(" ".join([subj, v, obj]))
        for token in doc:
            if not token.tag_ != 'NN' and token.tag_ != 'NNP' and token.text == subj:
                indices_2.append(idx)
        matched = False
        for v, rel in verbs_relations:
            # print(v, rel)
            if v in row["relation"]:
                verb = triples.at[idx, 'relation'].split(v)
                if len(verb) > 1 and verb[1] != "can":
                    triples.at[idx, 'object'] = verb[1] + " " + triples.at[idx, 'object']
                triples.at[idx, 'relation'] = rel
                matched = True
                break
        if not matched:
            if row["object"].startswith("because"):
                triples.at[idx, 'relation'] = "XReason"
            elif row["object"].startswith("so"):
                triples.at[idx, 'relation'] = "Causes"
            elif row["relation"] == "is":
                triples.at[idx, 'relation'] = "isA"
            else:
                indices_2.append(idx)
        if "d" in row["object"].split(" "):
            indices_2.append(idx)
        if "s" in row["object"].split(" "):
            indices_2.append(idx)

    triples.drop(triples.index[indices_2], inplace=True, axis=0)
    triples.drop(["Unnamed: 0", "length_s", "length_o"], axis=1, inplace=True)
    triples.to_csv("final_triplets.csv")

    print("Final triples", triples.shape)
    print(triples["relation"].value_counts())
    print("Splitting data....")
    shuffle_df = triples.sample(frac=1)
    N = triples.shape[0]
    data = {}
    data["train"] = shuffle_df[:int(N * 0.8)]
    data["val"] = shuffle_df[int(N * 0.8):int(N * 0.9)]
    data["test"] = shuffle_df[int(N * 0.9):]

    for split in ["train", "test", "val"]:
        print(split, data[split].shape)
        for idx, row in data[split].iterrows():
            head = row["subject"]
            rel = row["relation"]
            target = row["object"]
            with open(f"{split}.source", "a") as f:
                f.write("{} {} [GEN]".format(head, rel) + "\n")
            with open(f"{split}.target", "a") as f:
                f.write("{}".format(target) + "\n")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
s = set(stopwords.words('english'))


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _intersection(lst1, lst2):
    return len(set(lst1).intersection(lst2)) > 0


def prepare_data(exp_name, set_name, base_dir):
    # Load questions:
    annotations = _load_json(
        '/Users/sahiravi/Documents/Research/VL project/scratch/data/coco/aokvqa/aokvqa_v1p0_' + set_name + '.json')

    # Load expansions:
    expansions = _load_json(
        '/Users/sahiravi/Documents/Research/VL project/scratch/data/coco/aokvqa/commonsense/expansions/' + exp_name + '_aokvqa_' + set_name + '.json')

    # raw_expansions = _load_json(
    #     f'/Users/sahiravi/Documents/Research/VL project/scratch/data/coco/okvqa/commonsense/expansions/question_expansion_sentences_{set_name}_okvqa_{exp_name}.json')

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

        exps = expansions['{:012d}.jpg'.format(im_id)][str(q_id)][0]
        exps = exps.split('.')
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
    model = SentenceTransformer('msmarco-distilbert-base-tas-b')

    # Define your train examples. You need more than just two examples...
    train_egs = _load_json(train_file)
    train_examples = [InputExample(texts=e['sents'], label=e['label']) for e in train_egs]

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, output_path=base_dir)


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
