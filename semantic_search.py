"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer, util
import torch
import config
from PIL import Image

if torch.cuda.is_available():
    print("######## USING GPU ###############")
    device = 'cuda'
else:
    device = 'cpu'
# asymmetric
# pretrained = "msmarco-distilbert-base-v4"

# symmetric
# pretrained = "all-MiniLM-L6-v2"
# "multi-qa-MiniLM-L6-cos-v1"

# image symmetric
pretrained = "clip-ViT-B-32"

embedder = SentenceTransformer(pretrained, device=device)

def symmetric_search(queries, corpus, k=3, threshold=0):
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(k, len(corpus))
    result = []
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True, batch_size=max(128, len(corpus)), device=device)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        # dot_scores = util.dot_score(query_embedding, corpus_embeddings)[0].cpu()
        top_results = torch.topk(cos_scores, k=top_k)

        # print("\n\n======================\n\n")
        # print("Query:", query)
        # print(f"\nTop {k} most similar sentences in corpus:")
        sent = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score > threshold and sent not in result:
                sent.append(corpus[idx])

        result.append(" ".join(sent))
    return result


# Search that returns expansions closest to the query as well to the image & query intersection
def image_symmetric_search(image_id, queries, corpus, k=15, threshold=0):
    im_path = config.images_path + '/' + config.imageid_to_path(image_id)
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    image_embeddings = embedder.encode(Image.open(im_path), convert_to_tensor=True)

    top_k = min(k, len(corpus))

    # Find the closest 15 sentneces of the corpus to the image
    im_result = []
    cos_scores = util.pytorch_cos_sim(image_embeddings, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    for score, idx in zip(top_results[0], top_results[1]):
        if score > threshold and im_result not in im_result:
            im_result.append(corpus[idx])

    # Find the closest 15 sentences of the corpus for each query sentence based on cosine similarity
    qn_result = []
    result = []
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True, batch_size=max(128, len(corpus)), device=device)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        # dot_scores = util.dot_score(query_embedding, corpus_embeddings)[0].cpu()
        top_results = torch.topk(cos_scores, k=top_k)

        # print("\n\n======================\n\n")
        # print("Query:", query)
        # print(f"\nTop {k} most similar sentences in corpus:")
        sent = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score > threshold and sent not in qn_result:
                sent.append(corpus[idx])

        # Calculate the intersection of the image search results and the query search results
        result.append(" ".join(list(set(im_result) & set(sent))))
        
        qn_result.append(" ".join(sent))
    
    # print(qn_result)
    # print("\n\n======================\n\n")
    # print(result)

    return {'q': qn_result, 'qi': result}

def sentence_similarity(sentences1, sentences2):
    model = embedder

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Find the pairs with the highest cosine similarity scores
    pairs = []
    print(len(sentences1), len(sentences2))
    print(cosine_scores)
    for i in range(len(cosine_scores)):
        for j in range(len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    final = []
    for pair in pairs[0:5]:
        i, j = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], pair['score']))
        final.append(sentences1[i])
    return final


if __name__ == '__main__':

    # Corpus with example sentences
    corpus = ['A man is eating food.',
              'A man is eating a piece of bread.',
              'The girl is carrying a baby.',
              'A man is riding a horse.',
              'A woman is playing violin.',
              'Two men pushed carts through the woods.',
              'A man is riding a white horse on an enclosed ground.',
              'A monkey is playing drums.',
              'A cheetah is running behind its prey.',
              'A cop on a bike.',
              'A cop is riding a motorcycle.',
              'A car is behind the bush.',
              'Cop is located on the road.'
              ]
    # Query sentences:
    queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.',
               'A cheetah chases prey on across a field.', 'Who is riding the motorcycle?',
               'Is the driver in trouble?', 'What time of the day is it?']

    sentences1 = ['The cat sits outside',
                  'A man is playing guitar',
                  'The new movie is awesome']

    sentences2 = ['The dog plays in the garden',
                  'A woman watches TV',
                  'The new movie is so great']

    #symmetric_search(queries, corpus)
    image_symmetric_search('230008', queries, corpus, k=3)
    #sentence_similarity(sentences1, sentences2)