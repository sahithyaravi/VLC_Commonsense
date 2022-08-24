from tqdm import tqdm

from config import *
from question_to_phrases import QuestionConverter, remove_qn_words
from utils import load_json, qdict_to_df


def prepare(mthd, df, captions=None, object_tags=None, compress=False):
    logger.info("Converting caption expansions to sentences")
    question_converter = QuestionConverter()
    questions = list(df['question'].values)
    logger.info("Questions dataframe: ", df.head())
    print("Processing questions file......")
    qps = []
    for i in tqdm(range(len(questions))):
        question = questions[i]
        qp = question_converter.convert(question, compress)
        qps.append(qp)
    count = 0
    df['question_phrase'] = qps
    for idx, row in df.iterrows():
        q = row['question']
        qp = row['question_phrase']

        tokensq, tokensqp = question_converter.nlp(q), question_converter.nlp(qp)
        nounsq = [token.text for token in tokensq if token.pos_ == 'NOUN' or token.tag_ == 'PROPN']
        nounsqp = [token.text for token in tokensqp if token.pos_ == 'NOUN' or token.tag_ == 'PROPN']
        # print( len(nounsq), len(nounsqp))
        if not qp or len(nounsq) > len(nounsqp):
            df.at[idx, 'question_phrase'] = remove_qn_words(q.lower()).replace('?', '_').strip()
            # print(df.at[idx, 'question_phrase'])
            count += 1
    df["question_phrase"] = df["question_phrase"].str.replace("_", "")
    if mthd == "semcq":
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        df['question_caption_phrase'] = [(q + " and " + c.replace(".", "").lower()).capitalize() for q, c in
                                         zip(list(df["question_phrase"].values), captions)]

    else:
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        objs = ["with" + ",".join(object_tags[i]) for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_object_phrase'] = [c + o + " " + q for q, c, o in zip(qps, captions, objs)]


    print(f"{count}/{df.shape[0]}")
    df.to_csv(question_csv)


if __name__ == '__main__':
    # use config.py to configure dataset name, questions path etc
    print(questions_path)
    print(captions_path)
    caps = load_json(captions_path)
    # object_tags = load_json(objects_path)

    df = qdict_to_df(questions_path, dataset)
    prepare("semcq", df, caps)
