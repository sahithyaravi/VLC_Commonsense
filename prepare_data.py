from tqdm import tqdm

from config import *
from question_to_phrases import QuestionConverter, remove_qn_words
from utils import load_json, qdict_to_df
from joblib import Parallel, delayed

question_converter = QuestionConverter()


def convert_one_qn(question):
    return question_converter.convert(question)


def prepare(mthd, df, captions=None, object_tags=None, parallel=False):
    logger.info("Converting caption expansions to sentences")
    questions = list(df['question'].values)
    logger.info("Questions dataframe: ", df.head())
    print("Processing questions file......")

    if parallel:
        logger.info("Processing qn phrases in parallel")
        qps = zip(*Parallel(n_jobs=4)(
            delayed(convert_one_qn)(questions[i]) for i in
            tqdm(range(len(questions)))))
    else:
        qps = []
        for i in tqdm(range(len(questions))):
            q = questions[i]
            qps.append(convert_one_qn(q))
    count = 0
    df['question_phrase'] = qps
    zero_ents = 0
    for idx, row in df.iterrows():
        q = row['question']
        qp = row['question_phrase']

        tokensq, tokensqp = question_converter.nlp(q), question_converter.nlp(qp)
        nounsq = [token.text for token in tokensq if  token.tag_ == 'NN' or token.tag_ == 'NNP' or token.tag_ == "CD"]
        nounsqp = [token.text for token in tokensqp if token.tag_ == 'NN' or token.tag_ == 'NNP' or token.tag_ == "CD"]

        if len(nounsq) < 2:
            zero_ents += 1

        if len(nounsq) > len(nounsqp):
            df.at[idx, 'question_phrase'] = remove_qn_words(q.lower()).replace('?', '_').strip()
            count += 1
        df.at[idx, 'question_phrase'] = df.at[idx, 'question_phrase'].replace("best", "")
        df.at[idx, 'question_phrase'] = df.at[idx, 'question_phrase'].replace("describes", "")

    if mthd == "semcq":
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        df['question_caption_phrase'] = [(q + " and " + c.replace(".", "").lower()).capitalize() for q, c in
                                         zip(list(df["question_phrase"].values), captions)]

    if method == "semcqo":
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        objs = ["with" + ",".join(object_tags[i]) for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_object_phrase'] = [c + o + " " + q for q, c, o in zip(qps, captions, objs)]


    print(f"{count}/{df.shape[0]}")
    print(zero_ents)
    df["question_phrase"] = df["question_phrase"].str.replace("_", "")
    df.to_csv(question_csv)


if __name__ == '__main__':
    # use config.py to configure dataset name, questions path etc
    print(questions_path)
    print(captions_path)
    # caps = load_json(captions_path)
    # object_tags = load_json(objects_path)

    df = qdict_to_df(questions_path, dataset)
    prepare("semq", df)
