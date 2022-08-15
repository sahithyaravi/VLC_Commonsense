from tqdm import tqdm

from config import *
from question_to_phrases import QuestionConverter
from utils import load_json, qdict_to_df


def prepare(method, df, captions=None, object_tags=None):
    questions = list(df['question'].values)
    logger.info("Questions dataframe: ", df.head())
    print("Processing questions file......")
    qps = []
    for i in tqdm(range(len(questions))):
        question = questions[i]
        qp = question_converter.convert(question)
        qps.append(qp)
    if method == "sem-q":
        df['question_phrase'] = qps
        df.to_csv(question_csv)
    elif method == "sem-c":
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_phrase'] = [q + c for q, c in zip(qps, captions)]
        df.to_csv(question_csv)
    else:
        images_paths = list(df['image_path'].values)
        captions = [captions[i] for i in images_paths]
        objs = [object_tags[i] for i in images_paths]
        df['question_phrase'] = qps
        df['question_caption_object_phrase'] = [q + c + o for q, c, o in zip(qps, captions, objs)]
        df.to_csv(question_csv)


if __name__ == '__main__':
    logger.info("Converting caption expansions to sentences")
    question_converter = QuestionConverter()
    # use config.py to configure dataset name, questions path etc
    questions = load_json(questions_path)
    captions = load_json(captions_path)
    # object_tags = load_json(objects_path)

    df = qdict_to_df(questions)
    prepare("semq", df)
