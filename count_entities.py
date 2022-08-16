import spacy
import pandas as pd
from config import *

df = pd.read_csv(question_csv)
nlp = spacy.load('en_core_web_sm')
count = 0
for idx, row in df.iterrows():
    q = row['question']
    qp = row['question_phrase'].replace('_', '')
    tokensq, tokensqp = nlp(q), nlp(qp)
    nounsq = [token.text for token in tokensq if token.pos_=='NOUN']    
    nounsqp = [token.text for token in tokensqp if token.pos_=='NOUN']
    if len(nounsq) > len(nounsqp):
        # print(nounsq, nounsqp)
        count += 1

print(f"{count}/{df.shape[0]}")
