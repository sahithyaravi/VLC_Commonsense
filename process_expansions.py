import json
import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from config import *
from expansion_to_phrases import ExpansionConverter
from prepare_data import prepare
from semantic_search import symmetric_search
from utils import load_json, save_json, image_path_to_id, qdict_to_df

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def convert(image_originals, image_expansions, image_indicator, questions_df):
    convertor = ExpansionConverter()
    current_questions = questions_df[questions_df['image_path'] == image_indicator]
    qids = list(current_questions['question_id'].values)
    if current_questions.empty:
        return {}
    if type(image_originals) == dict:

        queries = list(current_questions['question_phrase'].values)

        all_sentences = []
        for i in range(len(queries)):
            out = convertor.convert(image_originals[qids[i]], image_expansions[qids[i]], False, queries[i])
            all_sentences.append(out)
        return dict(zip(qids, all_sentences))
    else:
        out = convertor.convert(image_originals, image_expansions, False)
        return dict(zip(qids, [out] * len(qids)))


def expansions_to_sentences(expansions, original_sentences, questions, save_path, srl=False, parallel=False,
                            use_cached=True):
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
        image_indicator = list(expansions.keys())
        if parallel:
            logger.info("Processing expansions in parallel")
            contexts, top_contexts = zip(*Parallel(n_jobs=-1)(
                delayed(convert)(original_sentences[image_indicator[i]], expansions[image_indicator[i]],
                                 image_indicator[i], questions_df) for i in
                tqdm(range(len(image_indicator)))))
        else:
            logger.warning("Processing expansions in series, set parallel=True to use multi-processing")
            contexts = []
            top_contexts = []
            for i in tqdm(range(len(image_indicator))):
                context, top_context = convert(original_sentences[image_indicator[i]], expansions[image_indicator[i]],
                                               srl, questions_df)
                contexts.append(context)
                top_contexts.append(top_context)
        all_contexts = dict(zip(image_indicator, contexts))
        save_json(save_path, all_contexts)
    return all_contexts


def search_expansions(expansions, questions_df, parallel=False):
    def single_image_semantic_search(img_path, image_expansion_sentences, qdf):
        picked_text = {}
        df_img = qdf[questions_df['image_path'] == img_path]
        if not df_img.empty:
            queries = list(df_img['question'].values)
            qids = list(df_img['question_id'].values)
            out, _ = [symmetric_search([queries[i]], image_expansion_sentences[qids[i]]) for i in range(len(qids))]
            picked_text = dict(zip(qids, out))
        return picked_text

    # we only need to process those images that have questions:
    question_image_ind = set(questions_df["image_path"].unique())
    image_indicators = list(question_image_ind & set(expansions.keys()))
    image_indicators.sort()
    if parallel:
        final_list = zip(*Parallel(n_jobs=-1)(
            delayed(single_image_semantic_search)(image_indicators[i], expansions[image_indicators[i]], questions_df)
            for i in tqdm(range(len(image_indicators)))))
        final_context = dict(zip(image_indicators, final_list))
    else:
        final_context = {}
        for i in tqdm(range(len(image_indicators))):
            text = single_image_semantic_search(image_indicators[i], expansions[image_indicators[i]], questions_df)
            final_context[str(image_indicators[i])] = text
    return final_context


if __name__ == '__main__':
    # load captions, questions and expansions
    captions = load_json(captions_path)
    object_tags = None
    caption_expansions = load_json(captions_comet_expansions_path)
    if os.path.exists(question_csv):
        questions_df = pd.read_csv(question_csv)
    else:
        questions_df = qdict_to_df(questions_path, dataset)

    logger.info("Number of questions: ", questions_df.shape[0])

    # Convert expansions to sentences
    logger.info("Converting caption expansions to sentences")

    if method == "sem-c":
        expansion_sentences = expansions_to_sentences(caption_expansions,
                                                      captions,
                                                      questions_df,
                                                      caption_expansion_sentences_path,
                                                      parallel=True)
    elif method == "sem-q":
        if "question_phrase" not in questions_df.columns:
            questions_df = prepare("sem-q", questions_df)
        question_phrases = list(questions_df["question_phrase"].values)
        question_expansions = load_json(questions_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(question_expansions,
                                                      questions_df,
                                                      question_phrases,
                                                      question_expansion_sentences_path,
                                                      parallel=True)
    elif method == "sem-cq":
        if "question_caption_phrase" not in questions_df.columns:
            questions_df = prepare("sem-cq", questions_df, captions)
        question_phrases = list(questions_df["question_caption_phrase"].values)
        question_expansions = load_json(questions_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(question_expansions,
                                                      questions_df,
                                                      question_phrases,
                                                      question_expansion_sentences_path,
                                                      parallel=True)
    elif method == "sem-cqo":
        if "question_caption_phrase" not in questions_df.columns:
            questions_df = prepare("sem-cqo", questions_df, captions, object_tags)
        question_phrases = list(questions_df["question_caption_object_phrase"].values)
        question_expansions = load_json(questions_comet_expansions_path)
        expansion_sentences = expansions_to_sentences(question_expansions,
                                                      questions_df,
                                                      question_phrases,
                                                      question_expansion_sentences_path,
                                                      parallel=True)
    else:
        expansion_sentences = {}

    final_results = search_expansions(expansion_sentences, questions_df, parallel=False)
    save_json(final_expansion_save_path + ".json", final_results)
