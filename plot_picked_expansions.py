import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

from config import *
from utils import imageid_to_path, load_json, save_json


def show_image(image_path, text="", title="", savefig_path="out.png"):
    fig = plt.figure()
    fig.suptitle(title)
    # plt.rcParams["figure.figsize"] = (25, 20)

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
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
    questions = load_json(questions_path)
    captions = load_json(captions_path)
    # Get questions as df
    df = pd.DataFrame(questions['questions'])
    df['image_id'] = df['image_id'].astype(str)
    df['question_id'] = df['question_id'].astype(str)
    picked_expansions = load_json(final_expansion_save_path +"I.json")
    print(len(picked_expansions))
    keys = list(picked_expansions.keys())
    print("Number of smaples", len(keys))

    for key in keys[:5]:
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
            text2 = ""
            texts.append(quest + "?\n" + text1 + "\n" + text2)
        show_image(image_path, "\n\n".join(texts), title=captions[filename])
        plt.show()

