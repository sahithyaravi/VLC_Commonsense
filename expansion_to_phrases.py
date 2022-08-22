import re
import string

import spacy
import textacy
from tqdm import tqdm

from config import *
from utils import load_json

relation_map = load_json("relation_map.json")


class ExpansionConverter:
    """
    Converting commonsense expansions to phrases.
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        self.srl_predictor = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        # object and subject constants
        self.OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}

        # Exclude these
        self.exclude_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
                             'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
                             'when was', 'when is',
                             'which is', 'which are', 'can you', 'which', 'would the',
                             'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']

        self.atomic_relations = ["oEffect",
                                 "oReact",
                                 "oWant",
                                 "xAttr",
                                 "xEffect",
                                 "xIntent",
                                 "xNeed",
                                 "xReact",
                                 "xReason",
                                 "xWant"]

        self.excluded_relations = [
            "CausesDesire",
            "DefinedAs",
            "DesireOf",
            "HasFirstSubevent",
            "HasLastSubevent",
            "HasPainCharacter",
            "HasPainIntensity",
            "HasSubEvent",
            "HasSubevent",
            "HinderedBy",
            "InheritsFrom",
            "InstanceOf",
            "IsA",
            "LocatedNear",
            "MotivatedByGoal",
            "NotCapableOf",
            "NotDesires",
            "NotHasA",
            "NotHasProperty",
            "NotIsA",
            "NotMadeOf",
            "ReceivesAction",
            "RelatedTo",
            "SymbolOf",
            "isFilledBy",
        ]

    def convert(self, sentence, exp, use_srl, question="", exclude_subject=False):
        """

        :param sentence: actual sentence provided as input to COMET
        :param exp: expansions of the sentences
        :param srl: if srl should be used for generating person x
        :return:
        """
        context = []
        top_context = []
        seen = set()
        excluded = [x.lower() for x in self.excluded_relations]
        personx = ""
        if question:
            # For negative questions, do not exclude not relations
            if "not" in question.split(" "):
                excluded.remove("notmadeof")
                excluded.remove("nothasproperty")
            personx_q = self.get_personx(question.replace("_", ""))

        # if not question or not personx:
        personx = self.get_personx(sentence.replace("_", ""))

        for relation, beams in exp.items():
            if relation.lower() not in excluded:
                top_context.append(
                    relation_map[relation.lower()].replace("{0}", personx).replace("{1}", beams[0]) + ".")
                for beam in beams:
                    source = personx
                    if beam != " none":
                        target = beam.lstrip().translate(str.maketrans('', '', string.punctuation))
                        if relation in self.atomic_relations and not self.is_person(source):
                            continue
                        if target and "none" not in target and target not in seen and not self.lexical_overlap(seen,
                                                                                                               target):
                            if exclude_subject:
                                sent = relation_map[relation.lower()].replace("{0}", "").replace("{1}", target)
                            else:
                                sent = relation_map[relation.lower()].replace("{0}", source).replace("{1}",
                                                                                                     target) + "."
                                sent = sent.capitalize()
                            context.append(sent)
                            seen.add(target)
        # print(context)
        return [context, top_context]

    def is_person(self, word):
        if word:
            living_beings_vocab = ["person", "people", "man", "woman", "girl", "boy", "child"
                                                                                      "bird", "cat", "dog", "animal",
                                   "insect", "pet"]
            refdoc = self.nlp(" ".join(living_beings_vocab))
            tokens = [token for token in self.nlp(word) if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
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

    def get_personx(self, input_event, use_chunk=True):
        """

        @param input_event:
        @param use_chunk:
        @return:
        """
        doc = self.nlp(input_event)
        personx = ""

        # Try subjects
        for token in doc:
            # print(token.dep_, token.text)
            if ("subj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return " ".join([a.text for a in doc[start:end]])
        # Try objects
        for token in doc:
            if ("dobj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return " ".join([a.text for a in doc[start:end]])

        # Try SVOS and noun phrases
        svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]
        if len(svos) == 0:
            if use_chunk:
                logger.info(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                # noun_chunks = [np.text
                #        for nc in doc.noun_chunks
                #        for np in [
                #            nc,
                #            doc[
                #            nc.root.left_edge.i
                #            :nc.root.right_edge.i + 1]]]
                # print(noun_chunks)

                if len(noun_chunks) > 0:
                    return noun_chunks[0]
                else:
                    logger.info("Didn't find noun chunks either, skipping this sentence.")
                    return ""

            else:
                logger.warning(
                    f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
                return ""

        else:
            subj_head = svos[0][0]
            personx = subj_head[0].text
            return personx

    def lexical_overlap(self, vocab, s1):
        if not vocab or not s1:
            return 0
        w1 = s1.split()

        for s2 in vocab:
            w2 = s2.split()
            overlap = len(set(w1) & set(w2)) / (len(w1)+1)
            if overlap > 0.7:
                return True
        return False

    def get_personx_srl(self, sentence):
        """

        @param sentence:
        @return:
        """
        from allennlp.predictors.predictor import Predictor
        predictor = Predictor.from_path(self.srl_predictor)
        results = predictor.predict(
            sentence=sentence
        )
        print(results)
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
        # print(personx)
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


if __name__ == '__main__':
    logger.info("Converting caption expansions to sentences")
    question_converter = ExpansionConverter()
    sample_questions = [
        "is in the motorcyclist 's mouth",
        "Number birthday is probably being celebrated ",
        "best describes the pool of water",
        "The white substance is on top of the cupcakes",
        "type of device is sitting next to the laptop",
        "A laptop computer sitting on top of a desk"]

    print("Checking sample questions")
    for i in tqdm(range(len(sample_questions))):
        question = sample_questions[i]
        print("IN: ", question)
        print("======================")
        qp = question_converter.get_personx_srl(question)
        print("SUBJ: ", qp)
        print("\n")
