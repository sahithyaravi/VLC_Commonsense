import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import spacy
import textacy
import logging
from semantic_search import symmetric_search

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')


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
                    # "CapableOf": "is capable of",
                    # "Desires": "desires",
                    "xIntent": "intended",
                    "xNeed": "needed to",
                    "xEffect": "then",
                    "xReact": "reacts",
                    "xWant": "wants",
                    "xAttr": "is seen as"}
    for key, exp in expansions.items():
        context = []
        personx, _ = get_personx(sentences[key])  # the sentence expanded by comet
        for relation, beams in exp.items():
            if relation in relation_map:
                context.append(personx + " " + relation_map[relation] + " " + beams[0])
        all_contexts[key] = context
    return all_contexts


def pick_expansions_method1(expanded_sentences, questions_df):
    final_context = {}
    for key, context in expanded_sentences.items():
        img_id = key.replace('COCO_train2014_000000', "")
        img_id = img_id.replace('.jpg', "")
        df_img = questions_df[questions_df['image_id'] == img_id]
        queries = df_img['question'].values
        picked_context = symmetric_search(queries, expansions, k=2)
        final_context[img_id] = picked_context
    return final_context




def show_image(image_path, text="", title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    # plt.rcParams["figure.figsize"] = (15, 15)
    plt.rcParams.update({'font.size': 10})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    ax2.text(0.1, 0.1, text, wrap=True)
    img = mpimg.imread(image_path)
    imgplot = ax1.imshow(img)


if __name__ == '__main__':
    # Open saved captions predictions, comet expansions for captions and questions
    captions_path = 'caption_expansions.json'
    comet_expansions_path = 'caption_comet_expansions.json'
    images_path = 'train2014'
    questions_path = 'v2_OpenEnded_mscoco_train2014_questions.json'

    with open(captions_path, 'r') as fp:
        captions = json.loads(fp.read())
    with open(comet_expansions_path, 'r') as fp:
        expansions = json.loads(fp.read())
    with open(questions_path, 'r') as fp:
        questions = json.loads(fp.read())

    # Get some sample images, expansions and questions and plot them
    df = pd.DataFrame(questions['questions'])
    df['image_id'] = df['image_id'].astype(str)
    image_groups = df.groupby('image_id')
    # for imgid, frame in image_groups:
    #     print(frame.head())
    caption_expansions_sentences = expansions_to_sentences(expansions, captions)
    # pick_expansions_method1(caption_expansions_sentences, questions)

    keys = list(captions.keys())
    for key in keys[30:40]:
        print(captions[key])
        # print(expansions[key])
        # print(caption_expansions_sentences[key])
        image_path = f'{images_path}/{key}'
        image_id = key.replace('COCO_train2014_000000', "")
        image_id = image_id.replace('.jpg', "")
        df_image = df[df['image_id'] == image_id]
        ques = "? ".join(df_image['question'].values)
        text = " . ".join(caption_expansions_sentences[key])
        show_image(image_path, ques + "\n\n" + text, title=captions[key])

        print(df_image)
        plt.show()
