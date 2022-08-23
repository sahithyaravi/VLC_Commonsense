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

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
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
    raw_expansions = load_json(question_expansion_sentences_path)

    # Get questions as df
    df = pd.read_csv(question_csv)
 
    for index, row in df.sample(50, random_state=33).iterrows():
        print(row)
        quest = row['question'] + "\n" + row["question_phrase"] + "\n" 
        qid = str(row['question_id'])
        img = row['image_path']
        image_path = f'{images_path}{img}'
        final_picked_expansions = ",".join(picked_expansions[img][qid])
        full_expansions = f"{raw_expansions[qid]} \n {row['direct_answers']}"
        text_input = ("\n\n\n" + quest + "\n" + full_expansions + "\n" + final_picked_expansions)
        show_image(image_path, text_input, title=captions[img], savefig_path=f"{index}_out.png")
        plt.show()