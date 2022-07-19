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

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.rcParams.update({'font.size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    ax2.text(0.1, 0.1, text, wrap=True)
    # if image_path:
    #     img = mpimg.imread(image_path)
    #     imgplot = ax1.imshow(img)
    # plt.show()
    fig.savefig(savefig_path)


if __name__ == '__main__':
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
    # captions = load_json(captions_path)
    # Get questions as df
    

    picked_expansions = load_json(final_expansion_save_path + ".json")
    print(len(picked_expansions))
    
    print(df['image_path'].values[0])
    keys = list(picked_expansions.keys())
    print(keys[0])
    print("Number of smaples", len(keys))

    c = 0
    for key in keys:
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
            show_image("", "\n\n".join(texts), title="", savefig_path=f"{c}_out.png")
            
            plt.show()

