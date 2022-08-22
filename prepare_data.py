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
    if mthd == "semq":
        df['question_phrase'] = qps

    elif mthd == "semcq":
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_phrase'] = [c + " " + q for q, c in zip(qps, captions)]

    else:
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        objs = ["with" + ",".join(object_tags[i]) for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_object_phrase'] = [c + o + " " + q for q, c, o in zip(qps, captions, objs)]

    count = 0
    for idx, row in df.iterrows():
        q = row['question']
        qp = row['question_phrase']


        tokensq, tokensqp = question_converter.nlp(q), question_converter.nlp(qp)
        nounsq = [token.text for token in tokensq if token.tag_ == 'NN' or token.tag_ == 'NNP']
        nounsqp = [token.text for token in tokensqp if token.tag_ == 'NN' or token.tag_ == 'NNP']
        if not qp or len(nounsq) > len(nounsqp):
            df.at[idx, 'question_phrase'] = remove_qn_words(q.lower()).replace('?', '').strip()
            # print(df.at[idx, 'question_phrase'])
            count += 1

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
