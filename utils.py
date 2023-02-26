import json
import re
import pandas as pd


from config import *




# helper functions
def imageid_to_path(image_id):
    n_zeros = 12 - len(str(image_id))
    if dataset == "aokvqa":
        filename = f'' + n_zeros * '0' + str(image_id) + '.jpg'
    else:
        filename = f'COCO_{split}2014_' + n_zeros * '0' + image_id + '.jpg'
    return filename


def image_path_to_id(image_fullname):
    img_id = image_fullname.replace(f'COCO_{split}2014_00', "")
    img_id = img_id.replace('.jpg', "")
    return str(int(img_id))


def load_json(filepath):
    with open(filepath, 'r') as fp:
        file = json.loads(fp.read())
    return file


def save_json(filename, data):
    with open(filename, 'w') as fpp:
        json.dump(data, fpp)





def qdict_to_df(questions_path, dataset):
    if dataset == "fvqa":
        qdict =  load_json(questions_path)
        df = pd.DataFrame.from_dict(qdict, orient='index')
        df['question_id'] = df['question_id'].astype(str)
        df["image_path"] = df["img_file"].astype(str)
    elif dataset == "vcr":
        questions = []
        with open(questions_path, 'r') as fp:
            for line in fp:
                questions.append(json.loads(line))
        df = pd.DataFrame(questions)
        df["image_id"] = df["img_id"]
        df["image_path"] = df["img_fn"]
        df["question_id"] = df["question_number"].astype(str)
        
        if "question_orig" not in list(df.columns):
            df["question_orig"] = df['question'].apply(lambda x: ' '.join(map(str, x)))
            df["question_orig"] = df["question_orig"].str.replace("[", "")
            df["question_orig"] = df["question_orig"].str.replace("]", "")

        df.drop("question", inplace=True, axis=1)
        df["question"] = df["question_orig"]
    else:
        qdict =  load_json(questions_path)
        if type(qdict) == list:
            df = pd.DataFrame(qdict)
            if "image_path" not in df.columns:
                ids = list(df["image_id"].values)
                df["image_path"] = [imageid_to_path(i) for i in ids]
        else:
            df = pd.DataFrame(qdict['questions'])
            df['image_id'] = df['image_id'].astype(str)
            df['question_id'] = df['question_id'].astype(str)
            paths = [imageid_to_path(k) for k in df["image_id"].values]
            df["image_path"] = paths

    return df




