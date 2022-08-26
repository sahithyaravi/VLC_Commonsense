import os.path
import re

import pandas as pd
import spacy
from openie import StanfordOpenIE
from tqdm import tqdm

# global variables

nlp = spacy.load('en_core_web_md')
stop_words = nlp.Defaults.stop_words
stop_words |= {"he", "she", "it", "they", "place", "kind", "type", "we", "they", "there"}
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


def is_ner(document):
    return len([(ent.text.strip(), ent.label_) for ent in document.ents]) > 0


def bad_subj(document, subj):
    for token in document:
        if token.text == subj and (token.tag_ != 'NN' and token.tag_ != 'NNP'):
            return True
    return False


def drop_all_duplicates(tdf):
    tdf.drop_duplicates(inplace=True, ignore_index=True, subset=['subject', 'object'])
    tdf.dropna(inplace=True, subset=['object'])
    tdf["length_o"] = tdf["object"].str.len()
    tdf = tdf.sort_values("length_o", ascending=False).drop_duplicates(subset=["subject", "relation"],
                                                                       keep="first")
    tdf["length_s"] = tdf["subject"].str.len()
    tdf = tdf.sort_values("length_s", ascending=False).drop_duplicates(subset=["object"],
                                                                       keep="first")
    tdf.reset_index(inplace=True, drop=True)
    return tdf


def filter_triples(tdf):
    indices = []
    # filtering
    for idx, row in tdf.iterrows():
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

    tdf.drop(tdf.index[indices], inplace=True, axis=0)
    tdf.reset_index(inplace=True, drop=True)
    return tdf


def map_relations_with_verbs(triples):
    for idx, row in triples.iterrows():
        subj = row["subject"]
        obj = row["object"].lower()
        v = row["relation"].lower()
        doc = nlp(" ".join([subj, v, obj]))
        for token in doc:
            if not token.tag_ != 'NN' and token.tag_ != 'NNP' and token.text == subj:
                continue
        matched = False
        for v, rel in verbs_relations:
            # print(v, rel)
            if v in row["relation"]:
                verb = triples.at[idx, 'relation'].split(v)
                if len(verb) > 1 and verb[1] != "can":
                    triples.at[idx, 'object'] = verb[1] + " " + triples.at[idx, 'object']
                triples.at[idx, 'relation'] = rel
                if rel == "CapableOf":
                    triples.at[idx, 'object'] = triples.at[idx, 'object'].replace("can", "being")
                matched = True
                break
        if not matched:
            if row["object"].startswith("because"):
                triples.at[idx, 'relation'] = "XReason"
            elif row["object"].startswith("so"):
                triples.at[idx, 'relation'] = "Causes"
            elif row["relation"] == "is":
                triples.at[idx, 'relation'] = "isA"

    triples["object"] = triples["object"].str.strip()
    triples["object"] = triples["object"].str.replace("d ", "")
    triples["object"] = triples["object"].str.replace("s ", "")

    triples.drop(["Unnamed: 0", "length_s", "length_o"], axis=1, inplace=True)
    return triples


def map_relations_with_phrases(triples):
    texts = []
    for idx, row in triples.iterrows():
        subj = row["subject"]
        obj = row["object"].lower()
        raw_rel = row["relation"].lower()
        text = " ".join([subj, raw_rel, obj])
        doc = nlp(text)
        if is_ner(doc) or bad_subj(doc, subj):
            continue
        texts.append(text)
        for v, rel in verbs_relations:
            if v in text.split(" "):
                parts = text.split(" " + v + " ")
                if len(parts) == 2:
                    head, rel, target = parts[0], rel, parts[1]
                    if v == "because":
                        head, rel, target = parts[1], rel, parts[0]
                    triples.at[idx, 'relation'] = rel
                    triples.at[idx, 'subject'] = head
                    triples.at[idx, 'object'] = target
                    break
    return triples


def train_val_test(finaldf):
    shuffle_df = finaldf.sample(frac=1)
    N = finaldf.shape[0]
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


def prepare_dataset_from_triplets(triples):
    triples = drop_all_duplicates(triples)
    triples = filter_triples(triples)
    triples.to_csv("svo_triples_filtered.csv")

    # map relations
    triples = map_relations_with_verbs(triples)
    triples = triples[triples.relation.isin([rel for v, rel in verbs_relations])]
    triples.to_csv("final_triplets.csv")

    print("Final triples", triples.shape)
    print(triples["relation"].value_counts())
    print("Splitting data....")

    train_val_test(triples)


def svo(text='Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'):
    """

    Parameters
    ----------
    text :

    Returns
    -------

    """
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
        'openie.triple.all_nominals': True,
        'openie.max_entailments_per_clause': 100
        # 'openie.resolve_coref': True,
    }
    with StanfordOpenIE(properties=properties) as client:
        print('Text: %s.' % text)
        return client.annotate(text)


def svo_openie(text='Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'):
    """

    Parameters
    ----------
    text :

    Returns
    -------

    """
    from allennlp.predictors import Predictor
    oie = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

    results = oie.predict(
        sentence=text
    )

    svos = []
    for v in results['verbs']:
        sub, rel, obj = '', '', ''
        text = v['description']
        search_results = re.finditer(r'\[.*?\]', text)
        outputs = {}
        for item in search_results:
            out = item.group(0).replace("[", "")
            out = out.replace("]", "")
            out = out.split(": ")
            if len(out) > 1:
                if 'ARGM' in out[0]:
                    outputs['ARGM'] = out[1]
                else:
                    outputs[out[0]] = out[1]
        # print(outputs)
        rel = v['verb']
        if 'ARG0' in outputs:
            if 'ARG1' in outputs:
                sub = outputs['ARG0']
                obj = outputs['ARG1']
            else:
                sub = outputs['ARG0']
                obj = outputs['ARGM'] if 'ARGM' in outputs else ''
        elif 'ARG1' in outputs:
            sub = outputs['ARG1']
            obj = outputs['ARGM'] if 'ARGM' in outputs else ''
        if sub and rel and obj:
            svos.append({'subject': sub, 'relation': rel, 'object': obj})

    return svos


def triplets(df, use_blacklist=True, stanford=True, save_path="comet/svo_triples.csv"):
    if os.path.exists(save_path):
        print("Reading ", save_path)
        all_triplets = pd.read_csv(save_path)
        prepare_dataset_from_triplets(all_triplets)
    else:
        directional = ['right', 'left', 'top', 'bottom', 'behind', 'under', 'inside', 'over', 'front', 'back', 'near',
                       'next', 'middle']
        other = ['logo', 'symbol', 'name', 'company', 'mascot', 'word', 'brand', "country", "many"]
        time = ['century', 'year', 'time', 'month', 'day']
        blacklist = directional + other + time

        if use_blacklist:
            for word in blacklist:
                df = df[~df.question.str.contains(word)]

        df["rationales_str"] = ["\n".join(map(str, l)) for l in df['rationales']]
        final = []
        # DO SVO extraction in batches of 500 using Stanford:
        if stanford:
            for i in range(0, 15000, 500):
                r = list(df["rationales_str"].values)[i:i + 500]
                rationale_list = "\n".join(r)
                svos = svo(rationale_list)
                final.extend(svos)
        else:
            with tqdm(total=df.shape[0]) as pbar:
                for idx, row in df.iterrows():
                    r = list(row["rationales"])
                    svos = []
                    for s in r:
                        trips = svo_openie(s)
                        # print(trips)
                        svos.extend(trips)
                    final.extend(svos)
                    pbar.update(1)

            all_triplets = pd.DataFrame(final)
            all_triplets.to_csv(save_path)
        prepare_dataset_from_triplets(all_triplets)



if __name__ == '__main__':
    # ROOT DIR
    root = "/Users/sahiravi/Documents/Research/VL project/scratch/data/coco"
    # root = "/ubc/cs/research/nlp/sahiravi/vlc_transformer/scratch/data/coco"
    method = "stanford"
    questions_path = f"{root}/aokvqa/aokvqa_v1p0_train.json"
    question_csv = pd.read_csv(questions_path.split("json")[0] + 'csv')
    if method == "stanford":
        triplets(question_csv)
    else:
        # Use ALLEN NLP instead
        triplets(question_csv, stanford=False)
