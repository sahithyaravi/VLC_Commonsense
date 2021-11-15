"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer, util
import torch


def symmetric_search(queries, corpus, k=1):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(k, len(corpus))
    result = []
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print(f"\nTop {k} most similar sentences in corpus:")
        sent = ""
        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))
            sent += corpus[idx] + "."
        result.append(sent)
    return result


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
          'A cheetah is running behind its prey.'
          ]
    # Query sentences:
    queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.',
               'A cheetah chases prey on across a field.']

    symmetric_search(queries, corpus)