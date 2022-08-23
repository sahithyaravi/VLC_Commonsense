import pandas as pd
from tqdm import tqdm
from question_to_phrases import QuestionConverter, remove_qn_words
from utils import load_json, qdict_to_df
import spacy
import textacy
from allennlp.predictors import Predictor

from openie import StanfordOpenIE

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}


def svo(text='Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'):
    with StanfordOpenIE(properties=properties) as client:
        print('Text: %s.' % text)
        return client.annotate(text)


    # triples_corpus = client.annotate("I am an idiot.")
    # print('Found %s triples in the corpus.' % len(triples_corpus))
    # for triple in triples_corpus[:3]:
    #     print('|-', triple)
    # print('[...]')
def triplets(df):
    relation_map = load_json("relation_map.json")
    keys = list(relation_map.keys())
    nlp = spacy.load('en_core_web_md')
    srl_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")


    training_set = []
    directional = ['right', 'left', 'top', 'bottom', 'behind', 'under', 'inside', 'over', 'front', 'back', 'near',
                   'next', 'middle']
    other = ['logo', 'symbol', 'name', 'company', 'mascot', 'word', 'brand', "country", "many"]
    time = ['century', 'year', 'time', 'month', 'day']
    blacklist = directional + other + time

    print(df.shape)
    for word in blacklist:
        df = df[~df.question.str.contains(word)]

    df = df[~df.question.str.isdigit()]
    df["rationales_str"] = ["\n".join(map(str, l)) for l in df['rationales']]
    final = []
    for i in range(0, 10000, 500):
        r = list(df["rationales_str"].values)[i:i+500]
        rationale_list = "\n".join(r)
        final.extend(svo(rationale_list))
    d = pd.DataFrame(final)
    d.to_csv("triples1.csv")

def eliminate_triples():
    triples = pd.read_csv("triples1.csv")
    print(triples.head())
    triples.drop_duplicates(inplace=True, ignore_index=True)
    triples.dropna(inplace=True, axis=0)
    print(triples.shape)


if __name__ == '__main__':
    questions_path = f'scratch/data/coco/aokvqa/aokvqa_v1p0_train.json'
    df = qdict_to_df(questions_path, "aokvqa")
    # triplets(df)
    eliminate_triples()

# results = srl_predictor.predict(
#     sentence=s
# )
# triplets = []
# for v in results['verbs']:
#     words = results['words']
#     tags = v['tags']
#
#     subj_words = " ".join([w for w, tag in zip(words, tags) if
#                  ('ARG0' in tag) ])
#     obj_words = " ".join([w for w, tag in zip(words, tags) if
#                  ('ARG1' in tag) ])
#     v_words = "".join([w for w, tag in zip(words, tags) if
#                  ('B-V' in tag) ])
#     triplets.append(subj_words+" "+obj_words+ " " + v_words)
# print(triplets)

# or ('ARG0' in tag) or ('ARG2' i
# n tag) or ('B-V' in tag)
# or ("ARGM-NEG" in tag))

# doc = nlp(s)
# svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]
# print(s)
# if len(svos) != 0:
#     subj_head = svos[0][0]
#     sub = " ".join(subj_head[i].text for i in range(len(subj_head)))
#     v_head = svos[0][1]
#     v = " ".join(v_head[i].text for i in range(len(v_head)))
#     o_head = svos[0][2]
#     o = " ".join(o_head[i].text for i in range(len(o_head)))
#
#     # print(s,v,o)
#     # if "can" in v and "see" not in v:
#     #     print(s)
#     #     # print(f"{sub} {v} {o}")