import pandas as pd
import spacy
from openie import StanfordOpenIE

from utils import load_json, qdict_to_df

# global variables
relation_map = load_json("relation_map.json")
nlp = spacy.load('en_core_web_md')
stop_words = nlp.Defaults.stop_words
stop_words |= {"he", "she", "it", "they", "place", "kind", "type"}


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
        'openie.max_entailments_per_clause': 3
        # 'openie.resolve_coref': True,
    }
    with StanfordOpenIE(properties=properties) as client:
        print('Text: %s.' % text)
        return client.annotate(text)


def triplets(df, use_blacklist=True):
    """

    Parameters
    ----------
    df : The dataframe with "rationales" column for which SVO triples need to be extracted
    use_blacklist: If we  need to omit brands, directions, and time based questions

    Returns
    -------

    """
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
    # DO SVO extraction in batches of 500
    for i in range(0, 15000, 500):
        r = list(df["rationales_str"].values)[i:i + 500]
        rationale_list = "\n".join(r)
        svos = svo(rationale_list)
        final.extend(svos)

    d = pd.DataFrame(final)
    d.to_csv("svo_triples.csv")


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
        ("want", "XWant"), # super bad
        ("was", "isBefore"),
        ("were", "isBefore"),
        ("will", "isAfter"),
        ("is for", "ObjectUse"),
        ("because", "Causes")
    ]

    for idx, row in triples.iterrows():
        subj = row["subject"]
        obj = row["object"].lower()
        v = row["relation"].lower()
        text = " ".join([subj, v, obj])
        doc = nlp(text)
        for token in doc:
            if not token.tag_ != 'NN' and token.tag_ != 'NNP' and token.text == subj:
                indices_2.append(idx)
        matched = False
        for v, rel in verbs_relations:
            if v in text:
                rel = verbs_relations[v]
                parts = text.split(rel)
                if len(parts) == 3:
                    head, rel, target = parts[0], rel, parts[2]
                    if v == "because":
                        head, rel, target = parts[2], rel, parts[0]
                    triples.at[idx, 'relation'] = rel
                    triples.at[idx, 'subject'] = head
                    triples.at[idx, 'object'] = target
                    matched = True
                    break
            if not matched:
                indices_2.append(idx)
        #     if row["object"].startswith("because"):
        #         triples.at[idx, 'relation'] = "XReason"
        #     elif row["object"].startswith("so"):
        #         triples.at[idx, 'relation'] = "Causes"
        #     elif row["relation"] == "is":
        #         triples.at[idx, 'relation'] = "isA"
        #     else:
        #
        # if "d" in row["object"].split(" "):
        #     indices_2.append(idx)
        # if "s" in row["object"].split(" "):
        #     indices_2.append(idx)

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

    # train.to_csv("train.tsv", header=False, index=False, sep="\t")
    # val.to_csv("val.tsv", header=False, index=False, sep="\t")
    # test.to_csv("test.tsv", header=False, index=False, sep="\t")


if __name__ == '__main__':
    questions_path = "/Users/sahiravi/Documents/Research/VL project/scratch/data/coco/aokvqa/aokvqa_v1p0_train.json"
    triplets(qdict_to_df(questions_path, "aokvqa"))
    refine_triples()

