import itertools
import re

import spacy
from allennlp.predictors import Predictor
from tqdm import tqdm

from config import *


logger = logging.getLogger(__name__)

Q_SENT = 'SBARQ'  # Question sentence
DETS = {'the', 'a', 'this', 'that', 'these', 'those'}
QUESTION_WORDS = {'what', 'where', 'who', 'when', 'why', 'how'}


class QuestionConverter:
    """
    Converting questions to (partial) declarative sentences.
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
        print(predictor.predict(
            sentence="If you bring $10 with you tomorrow, can you pay for me to eat too?."
        ))
        self.constituency_parser = predictor

    def convert(self, question):
        """
        Convert a question to a declarative sentence
        :param question: string question, possibly followed up by declarative sentences
        :return: a declarative sentence, possibly missing some words
        """
        doc = self.nlp(question)
        output = []
        spacy_tokens = []

        # Pre-processing: copy sentences or clauses without question mark.
        for sent in doc.sents:
            tokens = list(sent)

            # Declarative sentence
            if not tokens[-1].text == '?':
                output.append(' '.join([t.text for t in tokens]))
            # Contains a question
            else:
                spacy_tokens = tokens
                break

        if len(spacy_tokens) == 0:
            logger.warning(f'No question found in {question}')
            return question

        actual_question = ' '.join([t.text for t in spacy_tokens])
        const_parse = self.__constituents__(actual_question)
        capitalize_new_sentence = True

        # If the sentence contains a question but is not only a question, break it.
        if spacy_tokens[0].text.lower() not in QUESTION_WORDS:
            start_index = [i for i, token in enumerate(spacy_tokens) if token.text.lower() in QUESTION_WORDS]
            actual_questions = [' '.join([t.text for t in spacy_tokens[i:]]) for i in start_index]
            parses = [self.__constituents__(q) for q in actual_questions]
            results = [(i, q, p) for i, q, p in zip(start_index, actual_questions, parses) if p.node_type == Q_SENT]

            if len(results) > 0:
                i, actual_question, const_parse = results[-1]
                previous_tokens = [t.text for t in spacy_tokens[:i]]
                spacy_tokens = spacy_tokens[i:]
                output.append(' '.join(previous_tokens))
                capitalize_new_sentence = False

        # Associate each leaf in the parse tree with its spacy token
        const_parse = attach_constituency_with_spacy_tokens(const_parse, spacy_tokens)

        # Start applying rules. Each function returns None if the rule doesn't apply or the new
        # sentence if it does. Using python's short circuit, once one rule is applied successfully,
        # no other rules are applied.
        # The rule order is important - the more general rules should be in the end.
        new_sentence = what_np1_aux_np2_vp(const_parse) or what_as_subj(const_parse) or \
                       what_np1_aux_np2(const_parse) or qw_aux_np_vp(const_parse) or \
                       qw_anywhere(const_parse)

        # Couldn't convert to question - just remove the question mark
        if new_sentence is None:
            new_sentence = actual_question.replace('?', ' _').strip()
            new_sentence = remove_qn_words(new_sentence)

        # Capitalize the first word
        if capitalize_new_sentence:
            new_sentence = new_sentence.split()
            new_sentence = ' '.join([new_sentence[0].title()] + new_sentence[1:])

        output.append(new_sentence)
        sentence = ' '.join(output)
        logger.debug(sentence)
        return sentence

    def __constituents__(self, question):
        """
        Returns the constituency parse tree
        :param question: string question
        :return: the constituency parse tree
        """
        results = self.constituency_parser.predict(question)
        root = results['hierplane_tree']['root']
        root_node = build_tree(root)
        return root_node


def remove_qn_words(sent):
    # doc = nlp(sent)
    # for token in doc:
    #     print(token.text, ' ====> ', token.pos_)
    sent = sent.lower()
    substring_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
                      'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
                      'when was', 'when is', 'whose',
                      'which is', 'which are', 'can you', 'which', 'would the',
                      'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']

    for subs in substring_list:
        if sent.startswith(subs):
            sent = sent.replace(subs, "")
    return sent


def what_np1_aux_np2_vp(parse_tree):
    """
    What + NP1 + aux + NP2 + VP ->  The NP1 NP2 aux VP

    Examples:
    ========
    If you follow a road toward a river what feature are you driving through? =>
        If you follow a road toward a river the feature you are driving through
    What kind of service would you want if you do not want bad service or very good service?  =>
        The kind of service which you would want if you do not want bad service or very good service
    What kind of furniture would you put stamps in if you want to access them often?  =>
        The kind of furniture you would put stamps in if you want to access them often
    :param parse_tree: the constituency parse tree, associated to the spacy tokens
    :return: a new sentence if this rule applies, else None
    """
    rule_applies = parse_tree.node_type == Q_SENT and len(parse_tree.children) == 3 and \
                   parse_tree.children[0].node_type == 'WHNP' and \
                   len([child for child in parse_tree.children[0].children
                        if child.node_type in {'WHNP', 'NN', 'NNS', 'NNP', 'NNPS'}]) > 0 \
                   and parse_tree.children[1].node_type in {'S', 'SQ'} and \
                   len(parse_tree.children[1].children) > 2 and \
                   len(parse_tree.children[1].children[0].tokens) > 0 and \
                   (parse_tree.children[1].children[0].tokens[0].dep_ == 'aux' or \
                    parse_tree.children[1].children[0].tokens[0].lemma_ == 'be') and \
                   parse_tree.children[1].children[1].node_type == 'NP' and parse_tree.children[2].word == '?'

    if not rule_applies:
        return None

    question_phrase = parse_tree.children[0].word.lower().split()

    new_sentence = ['the' if question_phrase[1] not in DETS else '',
                    ' '.join(question_phrase[1:]),
                    parse_tree.children[1].children[1].word,
                    parse_tree.children[1].children[0].word,
                    ' '.join([node.word for node in parse_tree.children[1].children[2:]]),
                    'is _']
    new_sentence = ' '.join([const for const in new_sentence if const != ''])
    return new_sentence


def what_as_subj(parse_tree):
    """
    What + [aux] + VP/NP -> _ + [aux] + VP/NP

    Examples:
    ========
    What could be a serious consequence of typing too much? => _ could be a serious consequence of typing too much
    What might make a person stop driving to work and instead take the bus? =>
        _ might make a person stop driving to work and instead take the bus
    [If you're caught buying beer for children] what will happen? =>
        [If you're caught buying beer for children] _ will happen
    [After giving assistance to a person who's lost their wallet] what is customarily given? =>
        ..._ is customarily given

    What prevents someone from climbing?  =>  _ prevents someone from climbing
    What signals when an animal has received an injury? => _ signals when an animal has received an injury
    What leads someone to learning about world? => _ leads someone to learning about world

    :param parse_tree: the constituency parse tree, associated to the spacy tokens
    :return: a new sentence if this rule applies, else None
    """
    rule_applies = parse_tree.node_type == Q_SENT and parse_tree.word.lower().startswith('what') and \
                   len(parse_tree.children[0].children) <= 1 and \
                   len(parse_tree.children) == 3 and \
                   parse_tree.children[1].node_type in {'S', 'SQ'} and parse_tree.children[2].word == '?' and \
                   len(parse_tree.children[1].children) == 1 and \
                   parse_tree.children[1].children[0].node_type in {'NP', 'VP'}

    if not rule_applies:
        return None

    new_sentence = ['_', parse_tree.children[1].word]
    new_sentence = ' '.join([const for const in new_sentence if const != ''])
    return new_sentence


def what_np1_aux_np2(parse_tree):
    """
    What + NP1 + aux + NP2 -> The NP1 which aux NP2

    Examples:
    ========
    What events are typical, expected and not divine? => The events which are typical, expected and not divine are
    What mineral is plentiful in milk and helps bone development?  =>
        The mineral which is plentiful in milk and helps bone development is

    :param parse_tree: the constituency parse tree, associated to the spacy tokens
    :return: a new sentence if this rule applies, else None
    """
    rule_applies = parse_tree.node_type == Q_SENT and len(parse_tree.children) == 3 and \
                   parse_tree.children[0].node_type == 'WHNP' and \
                   len([child for child in parse_tree.children[0].children
                        if child.node_type in {'WHNP', 'NN', 'NNS', 'NNP', 'NNPS'}]) > 0 \
                   and parse_tree.children[1].node_type in {'S', 'SQ'} and \
                   len(parse_tree.children[1].children) == 1 and \
                   parse_tree.children[1].children[0].node_type == 'VP' and \
                   len(parse_tree.children[1].children[0].children) == 2 and \
                   len(parse_tree.children[1].children[0].children[0].tokens) > 0 and \
                   (parse_tree.children[1].children[0].children[0].tokens[0].dep_ == 'aux' or \
                    parse_tree.children[1].children[0].children[0].tokens[0].lemma_ == 'be') and \
                   parse_tree.children[1].children[0].children[1].node_type == 'ADJP' and \
                   parse_tree.children[2].word == '?'

    if not rule_applies:
        return None

    question_phrase = parse_tree.children[0].word.lower().split()

    new_sentence = ['the' if question_phrase[1] not in DETS else '',
                    ' '.join(question_phrase[1:]),
                    'which',
                    parse_tree.children[1].children[0].children[0].word,
                    parse_tree.children[1].children[0].children[1].word,
                    parse_tree.children[1].children[0].children[0].word,
                    '_']
    new_sentence = ' '.join([const for const in new_sentence if const != ''])
    return new_sentence


def qw_aux_np_vp(parse_tree):
    """
    QW + aux + NP + VP -> NP + aux + VP + question_word_to_preposition[QW]

    Examples:
    ========
    Where might Jenny be? => Jenny might be _
    Why would you be able to wait for someone? => You would be able to wait for someone because _
    When is food never cold?  =>  Food is never cold _
    How might a bank statement arrive at someone's house?  =>  A bank statement might arrive at someone's house _
    Who is someone competing against?  =>  Someone is competing against _
    What is a steel cable called a wire rope primarily used for?  =>
                                                    A steel cable called a wire rope is primarily used for _

    :param parse_tree: the constituency parse tree, associated to the spacy tokens
    :return: a new sentence if this rule applies, else None
    """
    question_word_to_preposition = {'where': '_', 'why': 'because _', 'when': '_', 'how': '_', 'who': '_', 'what': '_'}

    rule_applies = parse_tree.node_type == Q_SENT and \
                   len(parse_tree.children) == 3 and parse_tree.children[0].node_type.startswith('WH') and \
                   parse_tree.children[1].node_type in {'S', 'SQ'} and parse_tree.children[2].word == '?' and \
                   len(parse_tree.children[1].children) > 1 and \
                   len(parse_tree.children[1].children[0].tokens) > 0 and \
                   (parse_tree.children[1].children[0].tokens[0].dep_ == 'aux' or \
                    parse_tree.children[1].children[0].tokens[0].lemma_ == 'be') and \
                   parse_tree.children[1].children[1].node_type == 'NP'

    if not rule_applies:
        return None

    new_sentence = [parse_tree.children[1].children[1].word,  # NP
                    parse_tree.children[1].children[0].word if parse_tree.children[1].children[0].word != 'do' else '',
                    ' '.join([node.word for node in parse_tree.children[1].children[2:]]),  # VP / Adv / ...
                    question_word_to_preposition.get(parse_tree.children[0].word, '_')]  # prep

    new_sentence = ' '.join([const for const in new_sentence if const != ''])
    return new_sentence


def qw_anywhere(parse_tree):
    """
    Subj + verb + preposition + QW? -> Subj + verb + preposition + question_word_to_preposition[QW]

    Examples:
    ========
    A number is the usual response to what? => A number is the usual response to _
    There's an obvious prerequisite to being able to watch film, and that is to what? =>
        There's an obvious prerequisite to being able to watch film, and that is to _
    If you need to travel in the cold, you would be best to be what? =>
        If you need to travel in the cold, you would be best to be _
    He thinks that loving another will bring him what? =>
        He thinks that loving another will bring him _

    Bald eagles naturally live where?  =>  Bald eagles naturally live in _
    ...was his only connection to the outside world while doing time where? =>
        ...was his was his only connection to the outside world while doing time in _

    :param parse_tree: the constituency parse tree, associated to the spacy tokens
    :return: a new sentence if this rule applies, else None
    """
    question_word_to_preposition = {qw: '_' for qw in QUESTION_WORDS}
    question_word_to_preposition['how much'] = '_'
    question_word_to_preposition['how many'] = '_'
    question_word_to_preposition['where'] = 'in _'
    tokens = set(map(str.lower, parse_tree.word.split()))

    rule_applies = len(tokens.intersection(QUESTION_WORDS)) > 0

    if not rule_applies:
        return None

    # Some preference (how many/much replaces "how", "when" can be used not as a question)
    question_words = ['how many', 'how much'] + ['what'] + list(QUESTION_WORDS.difference({'what'}))
    new_sentence = parse_tree.word
    found = False

    for question_word in question_words:
        if question_word in tokens:
            new_sentence = re.sub(
                question_word, question_word_to_preposition[question_word], new_sentence, flags=re.IGNORECASE)
            found = True
            break

    if not found:
        return None

    return new_sentence.replace('?', '').strip()


def attach_constituency_with_spacy_tokens(const_parse, spacy_tokens):
    """
    Associate each leaf in the parse tree with its spacy token
    :param const_parse:
    :param spacy_tokens:
    :return:
    """
    spacy_tokens = list(spacy_tokens)
    spacy_index = 0
    leaves = [node for node in const_parse.post_order() if len(node.children) == 0]

    for const_index, node in enumerate(leaves):
        # Exact match
        if node.word == spacy_tokens[spacy_index].text:
            node.attach_spacy_tokens([spacy_tokens[spacy_index]])

        # Partial match - spacy is more split than the constituency parsing
        elif spacy_index < len(spacy_tokens) - 1 and node.word == ''.join(
                [t.text for t in spacy_tokens[spacy_index:spacy_index + 2]]):
            node.attach_spacy_tokens(spacy_tokens[spacy_index:spacy_index + 2])

        # Partial match - the constituency parsing is more split than spacy
        elif const_index < len(spacy_tokens) - 1 and spacy_tokens[spacy_index].text == ''.join(
                [n.word for n in leaves[const_index:const_index + 2]]):
            node.attach_spacy_tokens([spacy_tokens[spacy_index]])
            leaves[const_index + 1].attach_spacy_tokens([spacy_tokens[spacy_index]])

        spacy_index += 1
        if spacy_index >= len(spacy_tokens):
            break

    return const_parse


def build_tree(root):
    """
    Recursively build a tree from a dictionary
    :param root: the dictionary item corresponding to the root node
    :return: the root ParseTreeNode
    """
    curr = root
    curr_node = ParseTreeNode(root['word'], root['nodeType'])

    for child in curr.get('children', []):
        curr_node.add_child(build_tree(child))

    return curr_node


class ParseTreeNode:
    """
    A single node in a constituency parse tree, consisting of:

    - word
    - node_type: e.g. S, VP, NP, ...
    - children: list of ParseTreeNode objects
    """

    def __init__(self, word, node_type):
        self.word = word
        self.node_type = node_type
        self.children = []
        self.tokens = []

    def get_word(self):
        return self.word

    def get_node_type(self):
        return self.node_type

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def attach_spacy_tokens(self, tokens):
        self.tokens = tokens

    def get_spacy_token(self):
        return self.token

    def post_order(self):
        """
        Post-order tree traversal: traverse the children, then
        the parent.
        :return: a list of nodes.
        """
        if len(self.children) == 0:
            return [self]
        else:
            return list(itertools.chain(*[child.post_order() for child in self.children])) + [self]


if __name__ == '__main__':
    logger.info("Converting caption expansions to sentences")
    question_converter = QuestionConverter()
    sample_questions = ['Where might Jenny be?',
                        'Why would you be able to wait for someone?',
                        'When is food never cold?',
                        "How might a bank statement arrive at someone's house?",
                        'Who is someone competing against?',
                        'What is a steel cable called a wire rope primarily used for?',
                        'What could be a serious consequence of typing too much?',
                        'What might make a person stop driving to work and instead take the bus?',
                        "If you're caught buying beer for children what will happen?",
                        "After giving assistance to a person who's lost their wallet what is customarily given?",
                        "What is something I need to avoid while playing ball?",
                        "What is the opposite of being dead?", "What is the goal of a younger , risky investor?",
                        "James is a gardener. That is his profession. What is one thing that he cannot do in his job?",
                        "If you follow a road toward a river what feature are you driving through?",
                        "What kind of service would you want if you do not want bad service or very good service?",
                        "What kind of furniture would you put stamps in if you want to access them often?",
                        "What events are typical, expected and not divine?",
                        "What mineral is plentiful in milk and helps bone development?",
                        "What prevents someone from climbing?",
                        "What signals when an animal has received an injury?",
                        "What leads someone to learning about world?",
                        "A number is the usual response to what?",
                        "There's an obvious prerequisite to being able to watch film, and that is to what?",
                        "If you need to travel in the cold, you would be best to be what?",
                        "He thinks that loving another will bring him what?",
                        "It was his only connection to the outside world while doing time where?",
                        "It was tradition for the team to enter what through the central passage?",
                        "Fabric is cut to order at what type of seller?",
                        "A small dog will do well in what competition?",
                        "The wacky performer needed a new accordion, so he went where to get one?",
                        "Despite this name out front you will also find beer and wine where too?",
                        "What types of buildings typically have a column?",
                        "What is likely to be felt by someone after a kill?"]
    print("Checking sample questions")
    for i in tqdm(range(len(sample_questions[-5:]))):
        question = sample_questions[i]
        print(question)
        print("======================")
        qp = question_converter.convert(question)
        print(qp)
    # use config.py to configure dataset name, questions path etc
    # questions = load_json(questions_path)
    # df = qdict_to_df(questions)
    # questions = list(df['question'].values)
    # logger.info("Questions dataframe: ", df.head())
    # print("Processing questions file......")
    # qps = []
    # for i in tqdm(range(len(questions))):
    #     question = questions[i]
    #     qp = question_converter.convert(question)
    #     qps.append(qp)
    #
    # df['question_phrase'] = qps
    # df.to_csv(f'OpenEnded_mscoco_{split}_questions1.csv')
