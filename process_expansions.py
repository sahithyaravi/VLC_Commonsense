import os
import json
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import spacy
import textacy
import logging
from semantic_search import symmetric_search, sentence_similarity, image_symmetric_search
from config import *
# from allennlp.predictors.predictor import Predictor
from joblib import Parallel, delayed

os.environ["TOKENIZERS_PARALLELISM"] = "True"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')
exclude_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
                      'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
                      'when was', 'when is',
                      'which is', 'which are', 'can you', 'which', 'would the',
                      'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']

# srl_model_path = "https://storage.googleapis.com/allennlp-public-models/" \
#                  "structured-prediction-srl-bert.2020.12.15.tar.gz"
# srl_predictor = Predictor.from_path(srl_model_path)

def load_json(filepath):
    with open(filepath, 'r') as fp:
        file = json.loads(fp.read())
    return file


def save_json(filename, data):
    with open(filename, 'w') as fpp:
        json.dump(data, fpp)


def get_personx_srl(sentence):
    results = srl_predictor.predict(
        sentence=sentence
    )
    personx = {}
    for v in results['verbs']:
        # print(v['verb'])
        text = v['description']
        # print(text)
        search_results = re.finditer(r'\[.*?\]', text)
        for item in search_results:
            out = item.group(0).replace("[", "")
            out = out.replace("]", "")
            out = out.split(": ")
            # print(out)
            if len(out) >= 2:
                relation = out[0]
                node = out[1]
                if relation == 'ARG1' and v['verb'] not in personx:
                    personx[v['verb']] = node
                if relation == 'ARG0':
                    personx[v['verb']] = node
    print(personx)
    persons = list(personx.values())
    substring_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
                      'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
                      'when was', 'when is',
                      'which is', 'which are', 'can you', 'which', 'would the',
                      'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']
    returnval = persons[0] if persons else ""
    for subs in substring_list:
        if subs in returnval:
            returnval = returnval.replace(subs, "")
    
    return returnval if returnval else "person"


def get_personx(input_event, use_chunk=True):
    """
    get person x of the comet event
    :param input_event:
    :param use_chunk:
    :return:
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
        subj_head = svos[0][0]
        # is_named_entity = subj_head.root.pos_ == "PROP"
        personx = subj_head[0].text
        # " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])
    returnval = personx
    # substring_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
    #                   'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
    #                   'when was', 'when is',
    #                   'which is', 'which are', 'can you', 'which', 'would the',
    #                   'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']
    # returnval = personx
    # for subs in substring_list:
    #     if subs in returnval:
    #         returnval = returnval.replace(subs, "")

    return returnval if returnval else "This", False


def job(sentences, key, exp, srl):
    """

    :param sentences: actual sentence which was expanded
    :param key: index to identify sentences/expansions
    :param exp: expansions of the sentences
    :param srl: if srl should be used for generating person x
    :return:
    """
    context = []
    top_context = []
    relation_map = {
        "AtLocation": "is located at",
        "MadeUpOf": "is made of",
        "UsedFor": "is used for",
        "CapableOf": "can",
        "Desires": "desires",
        "NotDesires": "does not desire",
        "Causes": "causes",
        "HasProperty": "is",
        "xAttr": "is seen as",
        "xEffect": "then sees the effect",
        "xIntent": "wanted",
        "xNeed": "needed",
        "xReact": "feels",
        "xReason": "reasons",
        "xWant": "wants"}

    if srl:
        personx = get_personx_srl(sentences[key])
    else:
        personx, _ = get_personx(sentences[key])  # the sentence expanded by comet
    for relation, beams in exp.items():
        if relation in relation_map:
            top_context.append(personx + " " + relation_map[relation] + beams[0] + ".")
            for beam in beams[:2]:
                if beam != " none" and beam != "   ":
                    sent = personx + " " + relation_map[relation] + beam + "."
                    in_subs = False
                    for subs in exclude_list:
                        if sent.startswith(subs):
                            in_subs = True
                            break
                    if sent not in context and not in_subs:
                        context.append(sent)
    return context, top_context


def expansions_to_sentences(expansions, sentences, srl=False, parallel=False):
    if parallel:
        contexts, top_contexts = Parallel(n_jobs=-1)(
            delayed(job)(sentences, key, exp, srl) for key, exp in expansions.items())
    else:
        contexts = []
        top_contexts = []
        for key, exp in expansions.items():
            context, top_context = job(sentences, key, exp, srl)
            contexts.append(context)
            top_contexts.append(top_context)
    keys = list(expansions.keys())
    all_contexts = dict(zip(keys, contexts))
    all_top_contexts = dict(zip(keys, top_contexts))
    return all_contexts, all_top_contexts


def search_caption_expansions(caption_expanded, questions_df):
    """

    :param caption_expanded:
    :param questions_df:
    :return:
    """
    final_context = {}
    # print(qn_expansions_sentences.keys())
    i = 0
    for key, context in caption_expanded.items():
        i += 1
        # if i == 45:
        #     break
        img_id = image_path_to_id(key)
        df_img = questions_df[questions_df['image_id'] == img_id]
        if not df_img.empty:
            queries = list(df_img['question'].values)
            qids = list(df_img['question_id'].values)
            # picked = symmetric_search(queries, context, k=5, threshold=0.01)
            picked = image_symmetric_search(img_id, queries, context, k=15, threshold=0.01)
            image_dict = dict(zip(qids, picked))
            final_context[img_id] = image_dict
            if i % 10000 == 0:
                with open(f'picked{method}_{split}{i}.json', 'w') as fpp:
                    json.dump(final_context, fpp)
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
                if idx in question_expansions_sentences:
                    picked_context_qn = symmetric_search([qn], qn_expansions_sentences[idx], k=10, threshold=0.3)
                picked_context_caption = symmetric_search([qn], context, k=3, threshold=0)
                image_dict[idx] = picked_context_qn + picked_context_caption

                final_context[img_id] = image_dict
        if i % 10000 == 0:
            with open(f'picked{method}_{split}{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


def pick_expansions_method_top(top_question_exp, caption_expanded, questions_df):
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
            if idx in question_expansions_sentences:
                image_dict[idx] = " ".join(context) + " " + " ".join(top_question_exp[idx])
        final_context[img_id] = image_dict
        if i % 10000 == 0:
            with open(f'picked{method}_{split}{i}.json', 'w') as fpp:
                json.dump(final_context, fpp)
    return final_context


def show_image(image_path, text="", title=""):
    """

    :param image_path:
    :param text:
    :param title:
    :return:
    """
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


if __name__ == '__main__':
    # load captions, questions and expansions
    captions = load_json(captions_path)
    caption_expansions = load_json(captions_comet_expansions_path)
    question_expansions = load_json(questions_comet_expansions_path)
    questions = load_json(questions_path)

    # question dict to dataframe
    df = pd.DataFrame(questions['questions'])
    print(df.columns)
    df['image_id'] = df['image_id'].astype(str)
    df['question_id'] = df['question_id'].astype(str)

    # Expanded captions to sentences
    if os.path.exists(save_sentences_caption_expansions):
        caption_expansions_sentences = load_json(save_sentences_caption_expansions)
    else:
        caption_expansions_sentences, top_caption_expansions_sentences = expansions_to_sentences(caption_expansions,
                                                                                                 captions)
        save_json(f'{save_sentences_caption_expansions}', caption_expansions_sentences )
        save_json(f'{save_top_caption_expansions}', top_caption_expansions_sentences )

    # Pick final context using chosen method
    if method == "SEMANTIC_SEARCH":
        # This method only uses captions and picks a relevant caption expansion based on qn
        picked_expansions = search_caption_expansions(caption_expansions_sentences, df)

    else:
        # These two methods use both qn and caption expansions
        # Expand question expansions to sentences
        print("################### SENTENCES FROM QNS ################")
        if os.path.exists(save_sentences_question_expansions):
            with open(f'{save_sentences_question_expansions}', 'r') as fpp:
                question_expansions_sentences = json.loads(fpp.read())
        else:
            question_sentences = dict(zip(df.question_id, df.question))
            question_expansions_sentences, top_question_expansions_sentences = expansions_to_sentences(
                question_expansions, question_sentences, srl=False)
            save_json(f'{save_sentences_question_expansions}', question_expansions_sentences)
            save_json(f'{save_sentences_question_expansions}', top_question_expansions_sentences)

        if method == "SEMANTIC_SEARCH_QN":
            picked_expansions = search_caption_qn_expansions(question_expansions_sentences,
                                                             caption_expansions_sentences, df)
        elif method == "TOP":
            if os.path.exists(f'{save_top_qn_expansions}'):
                with open(f'{save_top_qn_expansions}', 'r') as fpp:
                    top_question_expansions_sentences = json.loads(fpp.read())
            if os.path.exists(f'{save_top_caption_expansions}'):
                with open(f'{save_top_caption_expansions}', 'r') as fpp:
                    top_caption_expansions_sentences = json.loads(fpp.read())

            picked_expansions = pick_expansions_method_top(top_question_expansions_sentences,
                                                           top_caption_expansions_sentences,
                                                           df)
            # picked_expansions = pick_expansions_method2(question_expansions_sentences,
            # caption_expansions_sentences, df)

    save_json(final_expansion_save_path, picked_expansions)

    # Plot first 5 final context + image samples
    keys = list(picked_expansions.keys())
    print("Number of smaples", len(keys))
    for key in keys:
        filename = imageid_to_path(key)
        image_path = f'{images_path}/{filename}'
        df_image = df[df['image_id'] == key]
        texts = []
        for index, row in df_image.iterrows():
            quest = row['question']
            qid = row['question_id']
            print(picked_expansions[key][qid])
            text = "".join(picked_expansions[key][qid])
            texts.append(quest + "?\n" + text)
        # print(caption_expansions[imageid_to_path(key)])
        # texts.extend(caption_expansions_sentences[filename])
        show_image(image_path, "\n\n".join(texts), title=captions[f'{imageid_to_path(key)}'])
        plt.show()
    #
