import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import pandas as pd

split = 'test2015'
images_path = f'data/vqa/{split}'
questions_path = f'data/vqa/questions/v2_OpenEnded_mscoco_{split}_questions.json'
captions_path = f'data/vqa/expansion/captions/captions_{split}_vqa.json'


def imageid_to_path(image_id):
    n_zeros = 12 - len(image_id)
    filename = f'COCO_{split}_' + n_zeros*'0' + image_id + '.jpg'
    return filename


def image_path_to_id(image_fullname):
    img_id = image_fullname.replace(f'COCO_{split}_00', "")
    img_id = img_id.replace('.jpg', "")
    return str(int(img_id))


def show_image(image_path, text="", title=""):
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
    # fig.savefig(f"{key}.jpg")


with open(questions_path, 'r') as fp:
    questions = json.loads(fp.read())
# Get questions as df
df = pd.DataFrame(questions['questions'])
df['image_id'] = df['image_id'].astype(str)
df['question_id'] = df['question_id'].astype(str)

with open('outputs/picked_expansions_SEMANTIC_SEARCHCAP_vqa_test2015_V2.json', 'r') as fp:
    picked_expansions = json.loads(fp.read())

keys = list(picked_expansions.keys())
print("Number of smaples", len(keys))
#
# output_refined = {}
# for key, dic in picked_expansions.items():
#     n_ques = (len(dic.keys()))
#     qids = list(dic.keys())
#     outs = []
#     for qid, lis in dic.items():
#         index = qids.index(qid)
#         print(len(lis))
#         outs.append(lis[-1]) if lis else outs.append("")
#     output_refined[key] = dict(zip(qids, outs))
#
# with open('outputs/picked_expansions_SEMANTIC_SEARCHCAP_vqa_test2015_V2.json', 'w') as fp:
#     json.dump(output_refined, fp)

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
        text = "".join(picked_expansions[key][qid])
        texts.append(quest + "?\n" + text)
    show_image(image_path, "\n\n".join(texts))
    plt.show()

