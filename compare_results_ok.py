import json
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils import load_json
from plot_picked_expansions import show_image


def get_count(res, answers):
    human_count = answers.count(res)
    return min(1.0, float(human_count) / 3)


if __name__ == '__main__':
    annotations = load_json('data/ok-vqa/mscoco_val2014_annotations.json')
    questions = load_json('data/ok-vqa/OpenEnded_mscoco_val2014_questions.json')
    results1 = load_json('compare/results_okvqa/19_semv1_okvqa_val2014.json')
    results2 = load_json('compare/results_okvqa/19_semv2_okvqa_val2014.json')
    captions = load_json('data/vqa/expansion/captions/captions_val2014_vqa.json')
    expansion = load_json('outputs/both-okvqa/picked_expansions_SEMANTIC_SEARCH_QN_ok-vqa_val2014_V2.json')

    ans_list = annotations['annotations']
    q_list = questions['questions']

    diffs = []
    total = len(results1)
    for i in range(len(results1)):
        if results1[i]['answer'] != results2[i]['answer']:
            diffs.append(i)

    random.seed(350)
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
        res['expansion'] = " ".join(expansion[image_id][str(q_id)])

        # image_id_to_path
        for k in range(0, 12 - len(image_id)):
            image_id = '0' + image_id
        img_path = "COCO_val2014_" + image_id + ".jpg"
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
        if res['state'] == 'bad':
            image_path = "data/vqa/val2014/" + res['image_path']
            title = res['question']
            text = res['caption'] + '\n' + res['expansion'] + '\n\nCaption+sem V1 Answer: ' + res[
                'answer_1'] + '\n Caption+semV2 Answer: ' + res['answer_2'] + '\n\n GT Answers: ' + ", ".join(
                res["possible_answers"])
            save_name = res['state'] + "-" + str(res['question_id'])
            show_image(image_path, text, title, save_name)

    print(len(diffs))
    print(total)
