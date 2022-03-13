import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

from config import *
from utils import imageid_to_path, load_json, save_json


def show_image(image_path, text="", title="", savefig_path="out.png"):
    plt.close('all')
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
    img = mpimg.imread(image_path)
    imgplot = ax1.imshow(img)
    # plt.show()
    fig.savefig(savefig_path)


if __name__ == '__main__':
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

    print(len(picked_expansions))
    keys = list(picked_expansions.keys())
    print("Number of smaples", len(keys))

    for key in keys[0:5]:
        filename = imageid_to_path(key)
        image_path = f'{images_path}/{filename}'
        print(key)
        df_image = df[df['image_id'] == key]
        print(df_image.head())
        texts = []
        for index, row in df_image.iterrows():
            quest = row['question']
            qid = row['question_id']
            text1 = picked_expansions[key][qid]
            norms = grad_norms_dict[int(qid)]
            text2 = str(norms)
            texts.append(quest + "?\n" + text1 + "\n" + text2)
        show_image(image_path, "\n\n".join(texts), title=captions[filename])
        plt.show()

