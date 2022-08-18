import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from config import *

from utils import load_json, save_json, image_path_to_id, qdict_to_df

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
    # plt.show()
    fig.savefig(savefig_path)


if __name__ == '__main__':
    picked_expansions = load_json(final_expansion_save_path + ".json")
    keys = list(picked_expansions.keys())
    print("Number of samples", len(keys))
    captions = load_json(captions_path)

    # Get questions as df
    df = pd.read_csv(question_csv)

    c = 0
    for key in keys[0:20]:
        filename = key
        image_path = f'{images_path}{filename}'
        df_image = df[df['image_path'] == key]
        if not df_image.empty:
            print(df_image.head())
            texts = []
            for index, row in df_image.iterrows():
                quest = row['question'] + "\n" + row["question_caption_phrase"]
                qid = row['question_id']
                if qid in (picked_expansions[key]):
                    text1 = ",".join(picked_expansions[key][qid])
                    texts.append(quest + "\n" + text1 + "\n")
                    c += 1
            show_image(image_path, "\n\n".join(texts), title=captions[key], savefig_path=f"{c}_out.png")
            plt.show()

