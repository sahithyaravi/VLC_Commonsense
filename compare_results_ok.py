import os
import random
import pandas as pd
from utils import load_json, image_path_to_id, imageid_to_path
from plot_picked_expansions import show_image
from config import *
# Don't forget to set val split and okvqa dataset in your config.py

def get_count(res, answers):
    human_count = answers.count(res)
    return min(1.0, float(human_count) / 3)


if __name__ == '__main__':
    # check if all the paths provided are correct:
    annotations = load_json(f'{data_root}/coco/annotations/mscoco_val2014_annotations.json')
    questions = load_json(f'{data_root}/coco/okvqa/OpenEnded_mscoco_val2014_questions.json')
    captions = load_json(f'{data_root}/coco/okvqa/commonsense/captions/captions_val2014_vqa.json')
    expansion = load_json(f'{data_root}/coco/okvqa/commonsense/expansions/sem1.3_okvqa_val2014.json')
    gpt3 = load_json(f'{data_root}/coco/okvqa/commonsense/gpt3/val2014_gpt3.json')
    grad_norms = load_json('result_files/okvqa/19_sem13_5_sbert_linear_prevqa_okvqa_val2014_gradnorms.json')
    grad_norms_df = pd.DataFrame(grad_norms)
    grad_norms_dict = dict(zip(grad_norms_df["question_id"].values, grad_norms_df["grad_norm"].values))

    # the two results to compare
    results1 = load_json('result_files/okvqa/caption_okvqa_val2014.json')
    results2 = load_json('result_files/okvqa/sem13_gpt3_5_sbert_linear_prevqa_okvqa_val2014.json')


    ans_list = annotations['annotations']
    q_list = questions['questions']

    # get all difference between results
    diffs = []
    total = len(results1)
    for i in range(len(results1)):
        if results1[i]['answer'] != results2[i]['answer']:
            diffs.append(i)

    # random 50 indices
    random.seed(42)
    for rand in range(50):
        rand_idx = random.randint(0, len(diffs) - 1)
        rand_idx = diffs[rand_idx]
        # print(results1[rand_idx])
        # print(results2[rand_idx])

        q_id = results1[rand_idx]["question_id"]

        ans = None
        for a in ans_list:
            if a['question_id'] == q_id:
                # print(a)
                ans = a
                break

        ques = None
        for q in q_list:
            if q['question_id'] == q_id:
                # print(q)
                ques = q
                image_id = q['image_id']
                break

        res = {}
        res['question_id'] = q_id
        res['image_id'] = image_id

        image_id = str(image_id)
        res['expansion'] = ",". join(expansion[image_id][str(q_id)].split(".")[:6])

        # image_id_to_path
        for k in range(0, 12 - len(image_id)):
            image_id = '0' + image_id
        img_path = imageid_to_path(image_id)
        caption = captions[img_path]
        # print(caption)

        res['image_path'] = img_path
        res['question'] = ques['question']
        res['caption'] = caption

        # first and second answer
        res['answer_1'] = results1[rand_idx]['answer']
        res['answer_2'] = results2[rand_idx]['answer']

        answers = [a['answer'] for a in ans['answers']]
        acc_res1 = get_count(res['answer_1'], answers)
        acc_res2 = get_count(res['answer_2'], answers)
        res['possible_answers'] = answers
        print(res['possible_answers'])

        if acc_res1 > acc_res2:
            res['state'] = 'bad'
        elif acc_res2 > acc_res1:
            res['state'] = 'good'
        else:
            res['state'] = 'neutral'

        # print(res)
        if res['state'] == 'good':
            image_path = f"{data_root}/coco/val2014/" + res['image_path']
            title = res['question']
            text = res['caption'] + '\n Context:' + res['expansion'] + '\n\n base Answer: ' + res[
                'answer_1'] + '\nimproved Answer: ' + res['answer_2'] + '\n\n GT Answers: ' + ", ".join(
                res["possible_answers"]) + str(grad_norms_dict[int(q_id)])
            print(text)
            print("gpt3", gpt3[str(res['question_id'])])

            # Save path - change this
            if not os.path.exists(res['state']):
                os.mkdir(res['state'])

            save_name = res['state'] + "/"+ str(res['question_id'])
            show_image(image_path, text, title, save_name)

    print(len(diffs))
    print(total)
