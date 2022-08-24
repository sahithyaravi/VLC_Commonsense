import pandas as pd
import spacy
from openie import StanfordOpenIE

from utils import load_json, qdict_to_df

# global variables
relation_map = load_json("relation_map.json")
nlp = spacy.load('en_core_web_md')


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
        # 'openie.resolve_coref': True,
        'openie.triple.all_nominals': True,
        # "annotators": "dcoref,tokenize,ssplit,pos,lemma,depparse,ner,coref,mention,natlog,openie"
        # 'openie.max_entailments_per_clause':1
    }
    with StanfordOpenIE(properties=properties) as client:
        print('Text: %s.' % text)
        return client.annotate(text)


def triplets(df):
    """
    Parameters
    ----------
    df :

    Returns
    -------

    """
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
    # # add a q_subject column
    # subjects = []
    # e = ExpansionConverter()
    # for idx, row in df.iterrows():
    #     subject = e.get_personx(row["question_phrase"])
    #     subjects.append(subject)
    # df["qsubject"] = subjects
    # # try to assign the same subject

    for i in range(0, 15000, 500):
        r = list(df["rationales_str"].values)[i:i + 500]
        rationale_list = "\n".join(r)
        svos = svo(rationale_list)
        print(svos)
        final.extend(svos)

    d = pd.DataFrame(final)
    d.to_csv("svo_triples.csv")


def refine_triples():
    """

    Returns
    -------

    """
    stop_words = nlp.Defaults.stop_words
    stop_words |= {"he", "she", "it", "they", "place", "kind", "type"}
    keys = list(relation_map.keys())
    triples = pd.read_csv("svo_triples.csv")
    triples.drop_duplicates(inplace=True, ignore_index=True, subset=['subject', 'object'])
    triples.dropna(inplace=True, subset=['object'])

    indices = []
    for idx, row in triples.iterrows():
        subj = row["subject"].lower()
        obj = row["object"].lower()
        if subj in stop_words:
            indices.append(idx)
        if len(obj) < 3 or len(subj) < 3:
            indices.append(idx)
        if "$" in subj:
            indices.append(idx)
        if obj in stop_words:
            indices.append(idx)
        if "answer" in row["object"].lower():
            indices.append(idx)

    triples.drop(triples.index[indices], inplace=True, axis=0)
    # triples = triples.groupby(['subject', 'object'], as_index=False).max()
    triples["length"] = triples["object"].str.len()
    triples = triples.sort_values("length", ascending=False).drop_duplicates(subset=["subject", "relation"],
                                                                             keep="first")
    triples.reset_index(inplace=True)
    triples.to_csv("svo_triples_filtered.csv")
    print(triples.shape)
    indices_2 = []
    verbs = ["so", "can", "could", "has", "have", "is in",
             "made", "consists", "part", "use",
             "uses", "used", "located",
             "is at", "is on", "always",
             "is behind", "is near",
             "is next to", "are behind", "are near",
             "are next to", "not has",
             "not", "don't", "is for", "want", "were",
             "was",  "will", "because"
             ]
    relations = [
        "Causes",
        "CapableOf",
        "CapableOf",
        "HasProperty",
        "HasProperty",
        "HasProperty",
        "MadeOf",
        "MadeOf",
        "PartOf",
        "UsedFor",
        "UsedFor",
        "UsedFor",
        "AtLocation",
        "AtLocation",
        "AtLocation",
        "HasProperty",
        "LocatedNear",
        "LocatedNear",
        "LocatedNear",
        "LocatedNear",
        "LocatedNear",
        "LocatedNear",
        "NotHasProperty",
        "NotHasProperty",
        "NotHasProperty",
        "ObjectUse",
        "XWant",
        "isBefore",
        "isBefore",
        "isAfter",
        "XReason"
    ]
    print(len(verbs), len(relations))
    for idx, row in triples.iterrows():
        matched = False
        for v, rel in zip(verbs, relations):
            # print(v, rel)
            if v in row["relation"]:
                verb = triples.at[idx, 'relation'].split(v)
                if len(verb) > 1:
                    triples.at[idx, 'object'] = verb[1] + triples.at[idx, 'object']
                triples.at[idx, 'relation'] = rel
                matched = True
                break
        if not matched:
            if row["object"].startswith("because"):
                triples.at[idx, 'relation'] = "XReason"
            elif row["object"].startswith("so"):
                triples.at[idx, 'relation'] = "Causes"
            elif row["object"].startswith("are"):
                triples.at[idx, 'relation'] = "isA"
            else:
                indices_2.append(idx)
            # elif row["object"].startswith(key):
            #     triples.at[idx, 'relation'] = value
            #     matched = True
            #     break

        # if not matched:
        #     if "is" in row["relation"]:
        #         verb = triples.at[idx, 'relation'].split(key)
        #         if len(verb) > 1:
        #             triples.at[idx, 'object'] = verb[1] + triples.at[idx, 'object']
        #         triples.at[idx, 'relation'] = value
        #     else:
        #         indices_2.append(idx)

        # elif "are" in row["relation"]:
        #     full_verb = triples.at[idx, 'relation'].split("is")
        #     triples.at[idx, 'relation'] = "IsA"
        #     if len(full_verb) > 1:
        #         triples.at[idx, 'object'] = full_verb[1] + " " + triples.at[idx, 'object']
        # triples.at[idx, 'object'] = triples.at[idx, 'object'].replace("can", "being")

    triples.drop(triples.index[indices_2], inplace=True, axis=0)
    triples.drop(["Unnamed: 0", "length"], axis=1, inplace=True)
    print(triples.columns)

    print(triples["relation"].nunique())
    print(triples.shape)
    triples.to_csv("converted_svos.csv")
    print(triples["relation"].value_counts())
    shuffle_df = triples.sample(frac=1)
    N = triples.shape[0]
    data = {}
    data["train"] = shuffle_df[:int(N * 0.8)]
    data["val"] = shuffle_df[int(N * 0.8):int(N * 0.9)]
    data["test"] = shuffle_df[int(N * 0.9):]

    for split in ["train", "test", "val"]:
        for idx, row in data[split].iterrows():
            head = row["subject"]
            rel = row["relation"]
            target = row["object"]
            with open(f"{split}.source", "a") as f:
                f.write("{} {} [GEN]".format(head, rel) + "\n")
            with open(f"{split}.target", "a") as f:
                f.write("{}".format(target) + "\n")

    # train.to_csv("train.tsv", header=False, index=False, sep="\t")
    # val.to_csv("val.tsv", header=False, index=False, sep="\t")
    # test.to_csv("test.tsv", header=False, index=False, sep="\t")


if __name__ == '__main__':
    questions_path = f'scratch/data/coco/aokvqa/aokvqa_v1p0_train.json'
    df = qdict_to_df(questions_path, "aokvqa")
    # triplets(df)
    refine_triples()

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
