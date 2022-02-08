import json
import os
import re

import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

from config import *
from semantic_search import symmetric_search, image_symmetric_search
from utils import get_personx, get_personx_svo, load_json, save_json, imageid_to_path, image_path_to_id

os.environ["TOKENIZERS_PARALLELISM"] = "True"
relation_map = load_json("relation_map.json")


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

    if srl:
        personx = get_personx_srl(sentences[key])
    else:
        personx = get_personx(sentences[key])  # the sentence expanded by comet
    for relation, beams in exp.items():
        top_context.append(relation_map[relation.lower()].replace("{0}", personx).replace("{1}", beams[0])+".")
        for beam in beams:
            if beam != " none" and beam != "   ":
                source = personx
                target = beam.lstrip()
                sent = relation_map[relation.lower()].replace("{0}", source).replace("{1}", target)+"."
                context.append(sent.capitalize())
    return context, top_context


def expansions_to_sentences(expansions, sentences, srl=False, parallel=False):
    print("Converting expansions to sentences:")
    keys = list(expansions.keys())[:5]
    if parallel:
        contexts, top_contexts = Parallel(n_jobs=-1)(
            delayed(convert_job)(sentences, keys[i], expansions[keys[i]], srl)for i in tqdm(range(len(keys))))
    else:
        contexts = []
        top_contexts = []
        for i in tqdm(range(len(keys))):
            context, top_context = convert_job(sentences, keys[i], expansions[keys[i]], srl)
            contexts.append(context)
            top_contexts.append(top_context)
    all_contexts = dict(zip(keys, contexts))
    all_top_contexts = dict(zip(keys, top_contexts))
    print(all_contexts)
    return all_contexts, all_top_contexts


def search_caption_expansions(caption_expanded, questions_df, parallel=True):
    def semantic_search_job(key, context, questions_df):
        out = {}
        img_id = image_path_to_id(key)
        df_img = questions_df[questions_df['image_id'] == img_id]
        if not df_img.empty:
            queries = list(df_img['question'].values)
            qids = list(df_img['question_id'].values)
            # picked = symmetric_search(queries, context, k=5, threshold=0.01)
            picked = image_symmetric_search(img_id, queries, context, k=15, threshold=0.01)
            image_dict = dict(zip(qids, picked))
            out[img_id] = image_dict
        return out

    keys = list(caption_expanded.keys())
    if parallel:
        final_context = Parallel(n_jobs=-1)(
            delayed(semantic_search_job)(keys[i], caption_expanded[keys[i]], questions_df) for i in tqdm(range(len(keys))))
    else:
        final_context = {}
        for i in tqdm(range(len(keys))):
            final = semantic_search_job(keys[i], caption_expanded[i], questions_df)
            final_context[keys[i]] = final

    return final_context


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

    # question dict to dataframe
    df = pd.DataFrame(questions['questions'])
    print(df.columns)
    df['image_id'] = df['image_id'].astype(str)
    df['question_id'] = df['question_id'].astype(str)

    # Expanded captions to sentences
    # If already cached, we load the sentences from the path
    if os.path.exists(save_sentences_caption_expansions):
        caption_expansions_sentences = load_json(save_sentences_caption_expansions)
    else:
        caption_expansions_sentences, top_caption_expansions_sentences = expansions_to_sentences(caption_expansions,
                                                                                                 captions)
        save_json(save_sentences_caption_expansions, caption_expansions_sentences)
        save_json(save_top_caption_expansions, top_caption_expansions_sentences)

    # Pick final context using chosen method
    if method == "SEMANTIC_SEARCH":
        # This method only uses captions and picks a relevant caption expansion based on qn
        picked_expansions = search_caption_expansions(caption_expansions_sentences, df)

    else:
        # These two methods use both qn and caption expansions
        print("Using quetion expansions")
        question_expansions = load_json(questions_comet_expansions_path)
        if os.path.exists(save_sentences_question_expansions):
            question_expansions_sentences = load_json(save_sentences_caption_expansions)
        else:
            question_sentences = dict(zip(df.question_id, df.question))
            question_expansions_sentences, top_question_expansions_sentences = expansions_to_sentences(
                question_expansions, question_sentences, srl=False)
            save_json(save_sentences_question_expansions, question_expansions_sentences)
            save_json(save_sentences_question_expansions, top_question_expansions_sentences)

        if method == "SEMANTIC_SEARCH_QN":
            picked_expansions = search_caption_qn_expansions(question_expansions_sentences,
                                                             caption_expansions_sentences, df)
        elif method == "TOP":
            if os.path.exists(save_top_qn_expansions):
                top_question_expansions_sentences = load_json(save_top_qn_expansions)
            if os.path.exists(save_top_caption_expansions):
                top_caption_expansions_sentences = load_json(save_top_caption_expansions)

            picked_expansions = pick_expansions_method_top(top_question_expansions_sentences,
                                                           top_caption_expansions_sentences,
                                                           df)
    save_json(final_expansion_save_path, picked_expansions)
