import os
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import spacy
import textacy
import logging
from semantic_search import symmetric_search, sentence_similarity
from config import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')


def imageid_to_path(image_id, split='train'):
    n_zeros = 12 - len(image_id)
    filename = f'COCO_{split}2014_' + n_zeros*'0' + image_id + '.jpg'
    return filename


def get_personx(input_event, use_chunk=True):
    """
    Returns the subject of a sentence
    We use this to get person x COMET has referred to
    """
    doc = nlp(input_event)
    svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]

    if len(svos) == 0:
        if use_chunk:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
            noun_chunks = [chunk for chunk in doc.noun_chunks]

            if len(noun_chunks) > 0:
                personx = noun_chunks[0].text
                is_named_entity = noun_chunks[0].root.pos_ == "PROP"
            else:
                logger.warning("Didn't find noun chunks either, skipping this sentence.")
                return "", False
        else:
            logger.warning(
                f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
            return "", False
    else:
        print(svos)
        subj_head = svos[0][0]
        print(subj_head)
        # is_named_entity = subj_head.root.pos_ == "PROP"
        personx = subj_head[
            0].text  # " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])

    return personx, False


def expansions_to_sentences(expansions, sentences):
    all_contexts = {}
    relation_map = {"AtLocation": "is located at",
                    "CapableOf": "is capable of",
                    "Causes": "causes",
                    "HasProperty": "has property",
                    "MadeOf": "is made of",
                    "NotCapableOf": "is not capable of",
                    "NotDesires": "does not desire",
                    "NotHasProperty": "does not have the property",
                    "NotMadeOf": "is not made of",
                    "RelatedTo": "is related to",
                    "UsedFor": "is used for",
                    "xAttr": "is seen as ",
                    "xEffect": "sees the effect",
                    "xIntent": "intends",
                    "xNeed": "needed to",
                    "xReact": "reacts",
                    "xReason": "reasons",
                    "xWant": "wants"}
    for key, exp in expansions.items():
        context = []
        personx, _ = get_personx(sentences[key])  # the sentence expanded by comet
        for relation, beams in exp.items():
            if relation in relation_map:
                for beam in beams:
                    if beam != " none":
                        context.append(personx + " " + relation_map[relation] + " " + beam)
        all_contexts[key] = context
    return all_contexts


def pick_expansions_method1(caption_expanded, questions_df):
    final_context = {}
    i = 0
    for key, context in caption_expanded.items():
        i += 1
        if i == 10:
            break
        img_id = key.replace('COCO_train2014_000000', "")
        img_id = img_id.replace('.jpg', "")
        df_img = questions_df[questions_df['image_id'] == img_id]
        queries = list(df_img['question'].values)
        if queries and context:
            # print(queries)
            # print(context)
            picked_context = symmetric_search(queries, context, k=3)
            image_dict = dict(zip(df_img['question_id'].values, picked_context))
            final_context[img_id] = image_dict
        if i % 1000 == 0:
            with open(f'picked{method}_train{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


def pick_expansions_method2(question_expansions_sentences, caption_expanded, questions_df):
    final_context = {}
    i = 0
    for key, context in caption_expanded.items():
        i += 1
        if i == 10:
            break
        img_id = key.replace('COCO_train2014_000000', "")
        img_id = img_id.replace('.jpg', "")
        df_img = questions_df[questions_df['image_id'] == img_id]
        queries = list(df_img['question'].values)
        qids = list(df_img['question_id'].values)
        image_dict = {}
        for idx in qids:
            if idx in question_expansions_sentences:
                sentb = question_expansions_sentences[idx]
                picked_context = sentence_similarity(sentb, context)
                image_dict[idx] = picked_context
        final_context[img_id] = image_dict
        if i % 1000 == 0:
            with open(f'picked{method}_train{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


def pick_expansions_method3(qn_expansions_sentences, caption_expanded, questions_df):
    final_context = {}
    print(qn_expansions_sentences.keys())
    i = 0
    for key, context in caption_expanded.items():
        i += 1
        if i == 20:
            break
        img_id = key.replace('COCO_train2014_000000', "")
        img_id = img_id.replace('.jpg', "")
        df_img = questions_df[questions_df['image_id'] == img_id]
        queries = list(df_img['question'].values)
        qids = list(df_img['question_id'].values)
        #print(qids)
        image_dict = {}
        for qn, idx in zip(queries, qids):
            if idx in qn_expansions_sentences:
                context.extend(qn_expansions_sentences[idx])
            picked_context = symmetric_search([qn], context, k=3)
            image_dict[idx] = picked_context
        final_context[img_id] = image_dict
        if i % 1000 == 0:
            with open(f'picked{method}_train{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


def show_image(image_path, text="", title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    # plt.rcParams["figure.figsize"] = (15, 15)
    plt.rcParams.update({'font.size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    ax2.text(0.1, 0.1, text, wrap=True)
    img = mpimg.imread(image_path)
    imgplot = ax1.imshow(img)
    # fig.savefig(f"{key}.jpg")


if __name__ == '__main__':
    # Open saved captions predictions, comet expansions for captions and questions
    with open(captions_path, 'r') as fp:
        captions = json.loads(fp.read())
    with open(captions_comet_expansions_path, 'r') as fp:
        caption_expansions = json.loads(fp.read())
    with open(questions_path, 'r') as fp:
        questions = json.loads(fp.read())
    with open(questions_comet_expansions_path, 'r') as fp:
        question_expansions = json.loads(fp.read())

    # Get questions as df
    df = pd.DataFrame(questions['questions'])
    df['image_id'] = df['image_id'].astype(str)
    df['question_id'] = df['question_id'].astype(str)
    # image_groups = df.groupby('image_id')
    # for imgid, frame in image_groups:
    #     print(frame.head())

    #   Expanded captions to sentences
    if os.path.exists(path_caption_expansions):
        with open(f'{path_caption_expansions}', 'r') as fpp:
            caption_expansions_sentences = json.loads(fpp.read())
            print("read expansions")
    else:
        caption_expansions_sentences = expansions_to_sentences(caption_expansions, captions)
        with open(f'{path_caption_expansions}', 'w') as fpp:
            json.dump(caption_expansions_sentences, fpp)

    if method == "SEMANTIC_SEARCH":
        picked_expansions = pick_expansions_method1(caption_expansions_sentences, df)

    else:
        if os.path.exists(path_question_expansions):
            with open(f'{path_question_expansions}', 'r') as fpp:
                question_expansions_sentences = json.loads(fpp.read())
        else:
            question_sentences = dict(zip(df.question_id, df.question))
            question_expansions_sentences = expansions_to_sentences(question_expansions, question_sentences)
            with open(f'{path_question_expansions}', 'w') as fpp:
                json.dump(question_expansions_sentences, fpp)
        # df = df.loc[df['question_id'].isin(list(question_expansions_sentences.keys()))]
        # print(df.head())
        if method == "SEMANTIC_SEARCH_QN":
            picked_expansions = pick_expansions_method3(question_expansions_sentences, caption_expansions_sentences, df)
        elif method == "SIMILARITY":
            picked_expansions = pick_expansions_method2(question_expansions_sentences, caption_expansions_sentences, df)

    with open(f'picked_expansions_{method}_train.json', 'w') as fpp:
        json.dump(picked_expansions, fpp)

    # Plot final output samples
    keys = list(picked_expansions.keys())
    for key in keys:
        filename = imageid_to_path(key)
        image_path = f'{images_path}/{filename}'
        # image_id = key.replace('COCO_train2014_000000', "")
        # image_id = image_id.replace('.jpg', "")
        df_image = df[df['image_id'] == key]
        texts = []
        for index, row in df_image.iterrows():
            quest = row['question']
            qid = row['question_id']
            text = " ".join(picked_expansions[key][qid])
            texts.append(quest+"?\n"+text)
        #texts.extend(caption_expansions_sentences[filename])
        show_image(image_path, "\n".join(texts), title=captions[f'COCO_train2014_000000{key}.jpg'])
        plt.show()
    #


