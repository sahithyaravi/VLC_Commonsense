import json
import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from config import *
from expansion_to_phrases import ExpansionConverter
# from prepare_data import prepare
from semantic_search import symmetric_search
from utils import load_json, save_json, image_path_to_id, qdict_to_df

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def convert(comet_input, original_expansion, expansion_key, q, method="semq"):
    # print(q.head())
    convertor = ExpansionConverter()
    if method == "semc":
        return convertor.convert(comet_input, original_expansion, False)
    else:
        current_df = q[q["question_id"] == str(expansion_key)]
        question = current_df["question_phrase"].values[0]
        return convertor.convert(comet_input,original_expansion, False, question)



def expansions_to_sentences(expansions, questions, original_sentences, save_path, srl=False, parallel=False,
                            use_cached=True, method='sem-q'):
    """
    Converts all the expansions from COMET into phrases or sentences using a template and text processing.

    Parameters
    ----------
    expansions : The expansions in structured format from COMET
    original_sentences : The original sentences provided to COMET
    save_path : Path for saving converted expansions
    srl : Use SRL to find person X or subject
    parallel : Generate sentences with multi-processing/in parallel
    use_cached : If set to True, this will use the already saved sentences file if present.
    questions: questions are used to decide the subject in some cases.

    Returns
    -------
    The expansions converted to sentences{id1:[list of sentences for id1].....}

    """
    if os.path.exists(save_path) and use_cached:
        logger.info(f"Using already cached sentences file: {save_path}")
        all_contexts = load_json(save_path)
    else:
        keys = list(expansions.keys())
        if parallel:
            logger.info("Processing expansions in parallel")
            contexts, top_contexts = zip(*Parallel(n_jobs=-1)(
                delayed(convert)(original_sentences[keys[i]], expansions[keys[i]], keys[i], questions, method) for i in tqdm(range(len(keys)))))
        else:
            logger.warning("Processing expansions in series, set parallel=True to use multi-processing")
            contexts = []
            top_contexts = []
            for i in tqdm(range(len(keys))):
                context, top_context = convert(original_sentences[keys[i]], expansions[keys[i]],
                                               srl, questions, method)
                contexts.append(context)
                top_contexts.append(top_context)
        all_contexts = dict(zip(keys, contexts))
        save_json(save_path, all_contexts)
    return all_contexts


def search_expansions(expansions, questions_df, parallel=False, method="semq"):
    def single_image_semantic_search(img_path, image_expansion_sentences, qdf):
        picked_text = {}
        df_img = qdf[questions_df['image_path'] == img_path]
        if not df_img.empty:
            queries = list(df_img['question'].values)
            qids = list(df_img['question_id'].values)
            if method == "semc":
                out = [symmetric_search([queries[i]], expansions[img_path]) for i in range(len(qids))]
            else:
                out = [symmetric_search([queries[i]], expansions[qids[i]]) for i in range(len(qids))]
            
            picked_text = dict(zip(qids, out))
        return picked_text

    # we only need to process those images that have questions:
    image_indicators = list(set(questions_df["image_path"].unique()))

    if parallel:
        final_list = zip(*Parallel(n_jobs=-1)(
            delayed(single_image_semantic_search)(image_indicators[i], expansions, questions_df)
            for i in tqdm(range(len(image_indicators)))))
        final_context = dict(zip(image_indicators, final_list))
    else:
        final_context = {}
        for i in tqdm(range(len(image_indicators))):
            text = single_image_semantic_search(image_indicators[i], expansions, questions_df)
            final_context[str(image_indicators[i])] = text
    return final_context



if __name__ == '__main__':
    # load captions, questions and expansions
    object_tags = None
    if os.path.exists(question_csv):
        questions_df = pd.read_csv(question_csv)
    else:
        questions_df = qdict_to_df(questions_path, dataset)

    logger.info("Number of questions: ", questions_df.shape[0])

    # Convert expansions to sentences
    logger.info("Converting caption expansions to sentences")

    if method == "semc":
        captions = load_json(captions_path)
        caption_expansions = load_json(captions_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(caption_expansions,
                                                      captions,
                                                      questions_df,
                                                      caption_expansion_sentences_path,
                                                      parallel=True,
                                                      method=method)
    elif method == "semq":
        questions_df["question_id"] = questions_df["question_id"].astype(str)
        # if "question_phrase" not in questions_df.columns:
        #     questions_df = prepare("sem-q", questions_df)
        question_phrases = dict(
            zip(list(questions_df["question_id"].values), list(questions_df["question_phrase"].values)))
        question_expansions = load_json(questions_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(question_expansions,
                                                      questions_df,
                                                      question_phrases,
                                                      question_expansion_sentences_path,
                                                      parallel=True,
                                                      method=method)
    elif method == "semcq":
        # if "question_caption_phrase" not in questions_df.columns:
        #     questions_df = prepare("semcq", questions_df, captions)
        question_phrases = dict(
            zip(list(questions_df["question_id"].values), list(questions_df["question_caption_phrase"].values)))
        question_expansions = load_json(cq_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(question_expansions,
                                                      questions_df,
                                                      question_phrases,
                                                      cq_expansion_sentences_path,
                                                      parallel=True,
                                                      method=method)
    elif method == "semcqo":
        captions = load_json(captions_path)
        caption_expansions = load_json(captions_comet_expansions_path)
        # if "question_caption_phrase" not in questions_df.columns:
        #     questions_df = prepare("semcqo", questions_df, captions, object_tags)
        question_phrases = dict(
            zip(list(questions_df["question_id"].values), list(questions_df["question_caption_object_phrase"].values)))
        question_expansions = load_json(questions_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(question_expansions,
                                                      questions_df,
                                                      question_phrases,
                                                      question_expansion_sentences_path,
                                                      parallel=True,
                                                      method=method)
    else:
        expansion_sentences = {}

    final_results = search_expansions(expansion_sentences, questions_df, parallel=False)
    save_json(final_expansion_save_path + ".json", final_results)
