import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import logging
import json
# Configure logging here
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure parameters for semantic search
# Methods to pick the final expansions
method = 'semq'  # [sem1- caption, sem2-caption+question]
version = '5'  # version of semantic search results
dataset = 'aokvqa'  # dataset 'vcr', 'okvqa' or 'aokvqa
data_root = "/Users/sahiravi/Documents/Research/VL project/scratch/data"  # root of the dataset folder arranged similar to VLC-BERT
data_root = "/ubc/cs/research/nlp/sahiravi/vlc_transformer/scratch/data"
split = 'val'


"""
@param: model_for_qn_search
"clip" - uses clip for both image and semantic search
"text" - uses clip for image search and text model for semantic search
"""

# Configure all paths here
if dataset == "okvqa":
    images_path = f'{data_root}/coco/{split}2014/'
    questions_path = f'{data_root}/coco/okvqa/OpenEnded_mscoco_{split}2014_questions.json'
    question_csv = questions_path.split("json")[0] + 'csv'
    captions_path = f'{data_root}/coco/okvqa/commonsense/captions/captions_{split}_vqa.json'
    captions_comet_expansions_path = f'{data_root}/coco/okvqa/commonsense/expansions/caption_comet_expansions_{split}_vqa_v3.json'
    questions_comet_expansions_path = f'{data_root}/coco/okvqa/commonsense/expansions/oquestion_comet_expansions_{split}_{dataset}.json'
    cq_comet_expansions_path = f'{data_root}/coco/okvqa/commonsense/expansions/cq_comet_expansions_{split}_{dataset}.json'
    caption_expansion_sentences_path = f'{data_root}/coco/okvqa/commonsense/expansions/caption_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    question_expansion_sentences_path = f'{data_root}/coco/okvqa/commonsense/expansions/question_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    cq_expansion_sentences_path = f'{data_root}/coco/okvqa/commonsense/expansions/cq_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    final_expansion_save_path = f'{data_root}/coco/okvqa/commonsense/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/coco/okvqa/commonsense/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/coco/okvqa/commonsense/expansions/top_qn_sentences_{split}_{dataset}.json'


elif dataset == 'vcr':
    images_path = f"{data_root}/vcr/vcr1images/"
    questions_path = f'{data_root}/vcr/{split}.jsonl'
    question_csv = questions_path.split("jsonl")[0] + 'csv'
    captions_path = f'{data_root}/vcr/commonsense/captions/captions_{split}_{dataset}.json'
    captions_comet_expansions_path = f'{data_root}/vcr/commonsense/expansions/caption_comet_expansions_{split}_{dataset}.json'
    questions_comet_expansions_path = f'{data_root}/vcr/commonsense/expansions/question_comet_expansions_{split}_{dataset}.json'
    caption_expansion_sentences_path = f'{data_root}/vcr/commonsense/expansions/caption_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    question_expansion_sentences_path = f'{data_root}/vcr/commonsense/expansions/question_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    final_expansion_save_path = f'{data_root}/vcr/commonsense/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/vcr/commonsense/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/vcr/commonsense/expansions/top_qn_sentences_{split}_{dataset}.json'
elif dataset == 'aokvqa':
    images_path = f'{data_root}/coco/{split}2017/'
    questions_path = f'{data_root}/coco/aokvqa/aokvqa_v1p0_{split}.json'
    question_csv = questions_path.split("json")[0] + 'csv'
    captions_path = f'{data_root}/coco/aokvqa/commonsense/captions/captions_{split}_{dataset}.json'
    captions_comet_expansions_path = f'{data_root}/coco/aokvqa/commonsense/expansions/caption_comet_expansions_{split}_{dataset}.json'
    questions_comet_expansions_path = f'{data_root}/coco/aokvqa/commonsense/expansions/oquestion_comet_expansions_{split}_{dataset}.json'
    cq_comet_expansions_path = f'{data_root}/coco/aokvqa/commonsense/expansions/cq_comet_expansions_{split}_{dataset}.json'
    caption_expansion_sentences_path = f'{data_root}/coco/aokvqa/commonsense/expansions/caption_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    question_expansion_sentences_path = f'{data_root}/coco/aokvqa/commonsense/expansions/question_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    cq_expansion_sentences_path = f'{data_root}/coco/aokvqa/commonsense/expansions/cq_expansion_sentences_{split}_{dataset}_{method}.{version}.json'
    final_expansion_save_path = f'{data_root}/coco/aokvqa/commonsense/expansions/{method}.{version}_{dataset}_{split}'
    topk_caption_path = f'{data_root}/coco/aokvqa/commonsense/expansions/top_cap_sentences_{split}_{dataset}.json'
    topk_qn_path = f'{data_root}/coco/aokvqa/commonsense/expansions/top_qn_sentences_{split}_{dataset}.json'
else:
    raise ValueError("Dataset must be one of okvqa, vcr or aokvqa")


def load_json(filepath):
    with open(filepath, 'r') as fp:
        file = json.loads(fp.read())
    return file

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
    # captions = load_json(captions_path)
    raw_expansions = load_json(question_expansion_sentences_path)

    # Get questions as df
    df = pd.read_csv(question_csv)

    for index, row in df.sample(10, random_state=33).iterrows():
        print(row)
        quest = row['question'] + "\n" + row["question_obj_phrase"] + "\n"
        qid = str(row['question_id'])
        img = row['image_path']
        image_path = f'{images_path}{img}'
        final_picked_expansions = ",".join(picked_expansions[img][qid])
        ans = row['direct_answers'] if 'direct_answers' in df.columns else ""
        full_expansions =  ""  #f"{raw_expansions[qid]} \n {ans}"
        text_input = ("\n\n" + "\n" + full_expansions + "\n" + final_picked_expansions)
        show_image(image_path, text_input, title=quest, savefig_path=f"{index}_out.png")
        plt.show()
