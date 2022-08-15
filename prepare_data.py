from tqdm import tqdm

from config import *
from question_to_phrases import QuestionConverter
from utils import load_json, qdict_to_df


def prepare(mthd, df, captions=None, object_tags=None):
    logger.info("Converting caption expansions to sentences")
    question_converter = QuestionConverter()
    questions = list(df['question'].values)
    logger.info("Questions dataframe: ", df.head())
    print("Processing questions file......")
    qps = []
    for i in tqdm(range(len(questions))):
        question = questions[i]
        qp = question_converter.convert(question)
        qps.append(qp)
    if mthd == "sem-q":
        df['question_phrase'] = qps
        df.to_csv(question_csv)
    elif mthd == "sem-cq":
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_phrase'] = [c + " " + q for q, c in zip(qps, captions)]
        df.to_csv(question_csv)
    else:
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        objs = ["with" + ",".join(object_tags[i]) for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_object_phrase'] = [c + o + " " + q for q, c, o in zip(qps, captions, objs)]
        df.to_csv(question_csv)


if __name__ == '__main__':

    # use config.py to configure dataset name, questions path etc
    print(questions_path)
    print(captions_path)
    caps = load_json(captions_path)
    # object_tags = load_json(objects_path)

    df = qdict_to_df(questions_path, dataset)
    prepare("sem-cq", df, caps)
