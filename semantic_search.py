import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

from config import *
from plot_picked_expansions import show_image
from utils import imageid_to_path

if torch.cuda.is_available():
    print("######## USING GPU ###############")
    device = 'cuda'
else:
    device = 'cpu'

# Define image and sentence embedder
image_model = "clip-ViT-B-32"
asymmetric_search_model = "msmarco-distilbert-base-v4"
image_embedder = SentenceTransformer(image_model, device=device)
if model_for_qn_search == "text":
    sentence_embedder = SentenceTransformer(asymmetric_search_model, device=device)
else:
    sentence_embedder = image_embedder


def symmetric_search(queries, corpus, k=15, threshold=0, return_as_lists=False):
    corpus_embeddings = sentence_embedder.encode(corpus, convert_to_tensor=True)
    top_k = min(k, len(corpus))
    result = []
    for query in queries:
        query_embedding = sentence_embedder.encode(query, convert_to_tensor=True,
                                                   batch_size=max(128, len(corpus)), device=device)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        # dot_scores = util.dot_score(query_embedding, corpus_embeddings)[0].cpu()
        top_results = torch.topk(cos_scores, k=top_k)
        sent = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score > threshold and sent not in result:
                sent.append(corpus[idx])
        if return_as_lists:
            result.append(sent)
        else:
            result.append(" ".join(sent))
    return result


# Search that returns expansions closest to the query as well to the image & query intersection
def image_symmetric_search(image_id, queries, corpus, k=15, threshold=0):
    im_path = images_path + '/' + imageid_to_path(image_id)
    show_image(im_path)
    corpus_embeddings = image_embedder.encode(corpus, convert_to_tensor=True)
    image_embeddings = image_embedder.encode(Image.open(im_path), convert_to_tensor=True)
    top_k = min(k*2, len(corpus))

    # Find the closest 15 sentneces of the corpus to the image
    im_result = []
    cos_scores = util.pytorch_cos_sim(image_embeddings, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    for score, idx in zip(top_results[0], top_results[1]):
        if score > threshold and im_result not in im_result:
            im_result.append(corpus[idx])

    # call question based semantic search - embedded using asymmetric model
    qn_results = symmetric_search(queries, corpus, k=100, return_as_lists=True)
    intersection_results = [set(r) & set(im_result) for r in qn_results]
    final_qn_results = [r[:k] for r in qn_results]
    return {'q': final_qn_results, 'qi': intersection_results}


if __name__ == '__main__':
    # Corpus with example sentences
    # set config to split = train2014
    if split == "train2014":
        corpus = ["A woman is located at a library.", "A woman is located at a room.",
                  "A woman is located at a building.",
                  "A woman is located at a hotel.", "A woman is located at clock.", "A woman is made of a clock.",
                  "A woman is made of the clock.", "A woman is made of a watch.", "A woman is made of a door.",
                  "A woman is made of a book.", "A woman is used for watch clock.", "A woman is used for watch tv.",
                  "A woman is used for a clock.", "A woman is used for watch time.", "A woman is a clock.",
                  "A woman is a mirror.", "A woman is a window.", "A woman is a book.",
                  "A woman is capable of watch the clock.", "A woman is capable of look at clock.",
                  "A woman is capable of watch the time.", "A woman is capable of see the time.",
                  "A woman is capable of watch clock.", "A woman desires to see the time.",
                  "A woman desires to see the clock.", "A woman desires watch the clock.",
                  "A woman desires watch the clock..", "A woman desires watch the time.",
                  "A woman does not desire to see the time.", "A woman does not desire to see the clock.",
                  "A woman does not desire not to be alone.", "A woman does not desire a clock..",
                  "A woman causes a clock.", "A woman causes a watch.", "A woman causes a book.",
                  "A woman causes watch.",
                  "A woman is seen as observant.", "A woman is seen as tired.", "A woman is seen as tired..",
                  "A woman is seen as a dancer.", "A woman is seen as a clock.",
                  "A woman sees the effect watches the clock.", "A woman sees the effect gets yelled at.",
                  "A woman sees the effect watches the time.", "A woman sees the effect gets a headache.",
                  "A woman sees the effect watches clock.", "A woman intends to see the time.",
                  "A woman intends to see the clock.", "A woman intends to get some rest.", "A woman intends a clock.",
                  "A woman needed to to look at the clock.", "A woman needed to to walk to the wall.",
                  "A woman needed to to look at the time.", "A woman needed to to look at the wall.",
                  "A woman needed to to have a clock.", "A woman reacts happy..", "A woman reacts tired..",
                  "A woman reacts tired.", "A woman reacts happy.", "A woman reacts relaxed.",
                  "A woman reasons a clock.",
                  "A woman reasons a watch.", "A woman reasons a book.", "A woman reasons watch.",
                  "A woman wants to look at the clock.", "A woman wants to look at the time.",
                  "A woman wants to check the time.", "A woman wants to take a nap.", "A woman wants to take a nap.."]

        # Query sentences:
        queries = ["Why is she getting ready to go and watching the clock at the same time?",
                   "What room is the picture likely taken?"]

        symmetric_search(queries, corpus)
        print(image_symmetric_search('277003', queries, corpus, k=15)["qi"])
