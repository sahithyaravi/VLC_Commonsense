import pandas as pd
import json
import spacy
nlp = spacy.load('en_core_web_sm')
pd.set_option('display.expand_frame_repr', False)
def remove_qn_words(sent):
    # doc = nlp(sent)
    # for token in doc:
    #     print(token.text, ' ====> ', token.pos_)
    sent = sent.lower()
    substring_list = ['what is', 'what are', 'where', 'where is', 'where are', 'what',
                      'how are', 'how many', 'how is', 'how', 'where is', 'where are', 'where',
                      'when was', 'when is',
                      'which is', 'which are', 'can you', 'which', 'would the',
                      'is the', 'is this', 'why did', 'why is', 'are the', 'do', 'why']

    for subs in substring_list:
        if sent.startswith(subs):
            sent = sent.replace(subs, "")
    return sent

questions_path = 'data/ok-vqa/OpenEnded_mscoco_val2014_questions.json'

with open(questions_path, 'r') as fp:
    questions = json.loads(fp.read())
# Get questions as df
df = pd.DataFrame(questions['questions'])
print(df.columns)
df['image_id'] = df['image_id'].astype(str)
df['question_id'] = df['question_id'].astype(str)
# image_groups = df.groupby('image_id')
# for imgid, frame in image_groups:
#     print(frame.head())

qns = []
for id, row in df.iterrows():
    qn = row['question']
    qn = qn.replace('?', '')
    qns.append(remove_qn_words(qn))

df['question_phrase'] = qns
print(df.head(50))
df.to_csv('OpenEnded_mscoco_val2014_questions.csv')