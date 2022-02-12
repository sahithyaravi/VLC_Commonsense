import json
import os
import string

import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

from config import *
from semantic_search import symmetric_search, image_symmetric_search
from utils import get_personx, load_json, save_json, image_path_to_id, is_person, qdict_to_df, lexical_overlap

os.environ["TOKENIZERS_PARALLELISM"] = "True"
relation_map = load_json("relation_map.json")
atomic_relations = ["oEffect",
                    "oReact",
                    "oWant",
                    "xAttr",
                    "xEffect",
                    "xIntent",
                    "xNeed",
                    "xReact",
                    "xReason",
                    "xWant"]

excluded_relations = ["causes", "xReason", "isFilledBy", "HasPainCharacter", "HasPainIntensity" ] # exclude some relations that are nonsense


def convert_job(sentences, key, exp, srl):
    """

    :param sentences: actual sentence which was expanded
    :param key: index to identify sentences/expansions
    :param exp: expansions of the sentences
    :param srl: if srl should be used for generating person x
    :return:
    """
    context = []
    top_context = []
    seen = set()

    if srl:
        personx = get_personx_srl(sentences[key])
    else:
        personx = get_personx(sentences[key])  # the sentence expanded by comet
    for relation, beams in exp.items():
        if relation not in excluded_relations:
            top_context.append(relation_map[relation.lower()].replace("{0}", personx).replace("{1}", beams[0])+".")
            for beam in beams:
                source = personx
                target = beam.lstrip().translate(str.maketrans('', '', string.punctuation))
                if relation in atomic_relations and not is_person(source):
                    source = "person"
                if beam and beam != "none" and target not in seen and not lexical_overlap(seen, target):
                    sent = relation_map[relation.lower()].replace("{0}", source).replace("{1}", target)+"."
                    context.append(sent.capitalize())
                    seen.add(target)

    return [context, top_context]


def expansions_to_sentences(expansions, sentences, save_path, topk_path, srl=False, parallel=False):
    print("Converting expansions to sentences:")
    all_top_contexts = {}
    if os.path.exists(save_path):
        all_contexts = load_json(save_path)
    else:

        keys = list(expansions.keys())
        if parallel:
            contexts, top_contexts = zip(*Parallel(n_jobs=-1)(
                delayed(convert_job)(sentences, keys[i], expansions[keys[i]], srl)for i in tqdm(range(len(keys)))))
        else:
            contexts = []
            top_contexts = []
            for i in tqdm(range(len(keys))):
                context, top_context = convert_job(sentences, keys[i], expansions[keys[i]], srl)
                contexts.append(context)
                top_contexts.append(top_context)
        all_contexts = dict(zip(keys, contexts))
        all_top_contexts = dict(zip(keys, top_contexts))
        save_json(save_path, all_contexts)
        save_json(topk_path, all_top_contexts)
    # print(all_contexts)
    return all_contexts, all_top_contexts


def search_caption_expansions(caption_expanded, questions_df, parallel=False):
    def semantic_search_job(img_path, context, questions_df):
        picked_img = {}
        picked_text = {}
        df_img = questions_df[questions_df['image_path'] == img_path]
        if not df_img.empty:
            queries = list(df_img['question'].values)
            qids = list(df_img['question_id'].values)
            # picked, _ = symmetric_search(queries, context, k=10, threshold=0.01)
            img_text, text_only = image_symmetric_search(img_path, queries, context, k=15, threshold=0)
            picked_img = dict(zip(qids, img_text))
            picked_text = dict(zip(qids, text_only))
        return picked_img, picked_text

    # we only need to process those images that have questions:
    question_image_ids = set(questions_df["image_path"].unique())
    caption_keys = set(caption_expanded.keys())
    img_paths = list(question_image_ids & caption_keys)
    img_ids = [image_path_to_id(key) for key in img_paths]

    if parallel:
        final_list_img, final_list = zip(*Parallel(n_jobs=4)(
            delayed(semantic_search_job)(img_paths[i], caption_expanded[img_paths[i]], questions_df) for i in tqdm(range(len(img_paths)))))
        final_context = dict(zip(img_ids, final_list))
        final_context_img = dict(zip(img_ids, final_list_img))
    else:
        final_context_img = {}
        final_context = {}
        for i in tqdm(range(len(img_paths))):
            img, text = semantic_search_job(img_paths[i], caption_expanded[img_paths[i]], questions_df)
            final_context_img[img_ids[i]] = img
            final_context[img_ids[i]] = text

    return final_context_img, final_context


def search_caption_qn_expansions(qn_expansions_sentences, caption_expanded, questions_df):
    """
    :param qn_expansions_sentences:
    :param caption_expanded:
    :param questions_df:
    :return:
    """
    final_context = {}
    # print(qn_expansions_sentences.keys())
    i = 0
    for key, context in caption_expanded.items():
        i += 1
        # if i == 300:
        #     break
        img_id = image_path_to_id(key)
        df_img = questions_df[questions_df['image_id'] == img_id]
        if not df_img.empty:
            queries = list(df_img['question'].values)
            qids = list(df_img['question_id'].values)
            image_dict = {}
            picked_context_qn = ""
            for qn, idx in zip(queries, qids):
                if idx in qn_expansions_sentences:
                    picked_context_qn = symmetric_search([qn], qn_expansions_sentences[idx], k=10, threshold=0.3)
                picked_context_caption = symmetric_search([qn], context, k=3, threshold=0)
                image_dict[idx] = picked_context_qn + picked_context_caption

                final_context[img_id] = image_dict
        if i % 10000 == 0:
            with open(f'picked{method}_{split}{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


def pick_expansions_method_top(top_question_exp, caption_expanded, questions_df, qn_expansion_sentences):
    """

    :param top_question_exp:
    :param caption_expanded:
    :param questions_df:
    :return:
    """
    final_context = {}
    # print(qn_expansions_sentences.keys())
    i = 0
    for key, context in caption_expanded.items():
        i += 1
        # if i == 5:
        #     break
        img_id = image_path_to_id(key)
        df_img = questions_df[questions_df['image_id'] == img_id]
        queries = list(df_img['question'].values)
        qids = list(df_img['question_id'].values)
        image_dict = {}
        picked_context1 = []
        for qn, idx in zip(queries, qids):
            if idx in qn_expansion_sentences:
                image_dict[idx] = " ".join(context) + " " + " ".join(top_question_exp[idx])
        final_context[img_id] = image_dict
        if i % 10000 == 0:
            with open(f'picked{method}_{split}{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


if __name__ == '__main__':
    # load captions, questions and expansions
    captions = load_json(captions_path)
    caption_expansions = load_json(captions_comet_expansions_path)
    questions = load_json(questions_path)
    df = qdict_to_df(questions)
    logger.info("Questions dataframe: ", df.head())

    logger.info("Converting caption expansions to sentences")

    caption_expansions_sentences, top_caption_expansions_sentences = expansions_to_sentences(caption_expansions,
                                captions, caption_expansion_sentences_path, topk_caption_path, parallel=True)

    logger.info(f"Starting to pick final expansions using {method}:")
    if method == "Sem_V1":
        out, out1 = search_caption_expansions(caption_expansions_sentences, df, parallel=False)

    elif method == "Sem_V2":
        question_expansions = load_json(questions_comet_expansions_path)
        question_expansions_sentences = expansions_to_sentences(question_expansions,
                                questions, question_expansion_sentences_path, topk_qn_path, parallel=False)
        out, out1 = search_caption_qn_expansions(question_expansions_sentences,
                                                             caption_expansions_sentences, df)
    else:
        logger.warning("You are not using semantic search. Picking using topk expansions")
        if os.path.exists(topk_qn_path):
            top_question_expansions_sentences = load_json(topk_qn_path)
        if os.path.exists(topk_caption_path):
            top_caption_expansions_sentences = load_json(topk_caption_path)

        out, out1 = pick_expansions_method_top(top_question_expansions_sentences,
                                                           top_caption_expansions_sentences,
                                                           df)
    save_json(final_expansion_save_path + "I.json", out)
    save_json(final_expansion_save_path + ".json", out1)
