import json
import re

import spacy
import textacy

from config import *

# Configure spacy models
nlp = spacy.load('en_core_web_md')

# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}

# Exclude these
exclude_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
                'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
                'when was', 'when is',
                'which is', 'which are', 'can you', 'which', 'would the',

                'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']


# helper functions
def imageid_to_path(image_id):
    n_zeros = 12 - len(image_id)
    filename = f'COCO_{split}_' + n_zeros * '0' + image_id + '.jpg'
    return filename


def image_path_to_id(image_fullname):
    img_id = image_fullname.replace(f'COCO_{split}_00', "")
    img_id = img_id.replace('.jpg', "")
    return str(int(img_id))


def load_json(filepath):
    with open(filepath, 'r') as fp:
        file = json.loads(fp.read())
    return file


def save_json(filename, data):
    with open(filename, 'w') as fpp:
        json.dump(data, fpp)


def is_person(word):
    living_beings_vocab = ["person", "people", "man", "woman", "girl", "boy", "child"
                           "bird", "cat", "dog", "animal", "insect", "pet"]
    refdoc = nlp(" ".join(living_beings_vocab))
    tokens = [token for token in nlp(word) if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
    avg = 0
    for token2 in tokens:
        for token in refdoc:
            sim = token.similarity(token2)
            if sim == 1:
                return True
            avg += sim
    avg = avg / len(refdoc)
    if avg > 0.5:
        return True
    return False


def get_personx(input_event, use_chunk=True):
    """

    @param input_event:
    @param use_chunk:
    @return:
    """
    doc = nlp(input_event)
    svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]

    if len(svos) == 0:
        if use_chunk:
            logger.info(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
            noun_chunks = [chunk for chunk in doc.noun_chunks]

            if len(noun_chunks) > 0:
                personx = noun_chunks[0].text
                # is_named_entity = noun_chunks[0].root.pos_ == "PROP"
                return personx
            else:
                logger.info("Didn't find noun chunks either, skipping this sentence.")
                return ""
        else:
            logger.warning(
                f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
            return ""
    else:
        subj_head = svos[0][0]
        # is_named_entity = subj_head[0].root.pos_ == "PROP"
        personx = subj_head[0].text
        # " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])
        return personx


def get_personx_svo(sentence):
    """

    @param sentence:
    @return:
    """
    doc = nlp(sentence)
    sub = []
    at = []
    ve = []
    for token in doc:
        # is this a verb?
        if token.pos_ == "VERB":
            ve.append(token.text)
        # is this the object?
        if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
            at.append(token.text)
        # is this the subject?
        if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
            sub.append(token.text)
    personx = sub[0]
    return personx


def get_personx_srl(sentence):
    """

    @param sentence:
    @return:
    """
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


def test_personx():
    s1 = "A man eating a chocolate covered donut with sprinkles."
    s2 = "A desk with a laptop, monitor, keyboard and mouse."
    print(get_personx(s1))
    print(get_personx(s2))


