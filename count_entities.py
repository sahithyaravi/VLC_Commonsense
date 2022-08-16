import pandas as pd
import spacy

from config import *
from question_to_phrases import remove_qn_words
df = pd.read_csv(question_csv)
nlp = spacy.load('en_core_web_md')
count = 0
for idx, row in df.iterrows():
    q = row['question']
    qp = row['question_phrase']
    tokensq, tokensqp = nlp(q), nlp(qp)
    nounsq = [token.text for token in tokensq if token.tag_ == 'NN' or token.tag_ == 'NNP']
    nounsqp = [token.text for token in tokensqp if token.tag_ == 'NN' or token.tag_ == 'NNP']
    if len(nounsq) > len(nounsqp):
        print(nounsq, nounsqp)
        count += 1

print(f"{count}/{df.shape[0]}")
