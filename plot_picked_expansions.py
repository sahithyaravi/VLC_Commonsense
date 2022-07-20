import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

from config import *

from utils import get_personx, load_json, save_json, image_path_to_id, is_person, qdict_to_df, lexical_overlap

def show_image(image_path="", text="", title="", savefig_path="out.png"):
    fig = plt.figure()
    fig.suptitle(title)
    # plt.rcParams["figure.figsize"] = (25, 20)

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.rcParams.update({'font.size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    ax2.text(0.1, 0.1, text, wrap=True)
    if image_path:
        img = mpimg.imread(image_path)
        imgplot = ax1.imshow(img)
    plt.show()
    fig.savefig(savefig_path)


if __name__ == '__main__':
<<<<<<< HEAD
    if dataset == "vcr":
        data = []
        with open(questions_path, 'r') as fp:
            for line in fp:
                data.append(json.loads(line))

        df = pd.DataFrame(data)
        df["image_id"] = df["img_id"]
        df["image_path"] = df["img_fn"]
        df["question_id"] = df["question_number"].astype(str)
        df.drop("question", inplace=True, axis=1)
        df["question"] = df["question_orig"]
        df = df.sample(1000)
    else:
        questions = load_json(questions_path)
        df = qdict_to_df(questions, dataset)
    captions = load_json(captions_path)
    # Get questions as df
    

    picked_expansions = load_json(final_expansion_save_path + ".json")
=======
    annotations = load_json(f'{data_root}/ok-vqa/mscoco_val2014_annotations.json')
    questions = load_json(f'{data_root}/ok-vqa/OpenEnded_mscoco_val2014_questions.json')
    captions = load_json(f'{data_root}/vqa/expansion/captions/captions_val2014_vqa.json')
    picked_expansions = load_json('final_outputs/okvqa/sem1.3/sem1.3_okvqa_val2014.json')
    gpt3 = load_json('final_outputs/gpt3/val2014_gpt3.json')
    grad_norms = load_json('eccv_results/19_sem13_5_sbert_linear_prevqa_okvqa_val2014_gradnorms.json')
    grad_norms_df = pd.DataFrame(grad_norms)
    grad_norms_dict = dict(zip(grad_norms_df["question_id"].values, grad_norms_df["grad_norm"].values))


    # Get questions as df
    df = pd.DataFrame(questions['questions'])
    df['image_id'] = df['image_id'].astype(str)
    df['question_id'] = df['question_id'].astype(str)

>>>>>>> e4278a6fd017b50bf6462580fc569faf4ba35425
    print(len(picked_expansions))
    
    print(df['image_path'].values[0])
    keys = list(picked_expansions.keys())
    print(keys[0])
    print("Number of smaples", len(keys))

    c = 0
    for key in keys[100:200]:
        filename = key
        image_path = f'{images_path}/{filename}'
        df_image = df[df['image_path'] == key]
        if not df_image.empty:
            print(df_image.head())
            texts = []
            for index, row in df_image.iterrows():
                quest = row['question']
                qid = row['question_id']
                if qid in (picked_expansions[key]):
                    text1 = picked_expansions[key][qid]
                    text2 = ""
                    texts.append(quest + "?\n" + text1 + "\n" + text2)
                    c += 1
            show_image(image_path, "\n\n".join(texts), title=captions[key], savefig_path=f"{c}_out.png")
            
            plt.show()

