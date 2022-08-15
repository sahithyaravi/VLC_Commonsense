import os
import random
from utils import load_json, imageid_to_path
from analysis.plot_picked_expansions import show_image
from config import *

def get_count(res, answers):
    human_count = answers.count(res)
    return min(1.0, float(human_count) / 3)


if __name__ == '__main__':
    # check if all the paths provided are correct:
    annotations = load_json(f'{data_root}/coco/aokvqa/aokvqa_v1p0_val.json')
    questions = load_json(f'{data_root}/coco/aokvqa/aokvqa_v1p0_val.json')
    captions = load_json(f'{data_root}/coco/aokvqa/commonsense/captions/captions_val_aokvqa.json')
    expansion = load_json(f'{data_root}/coco/aokvqa/commonsense/expansions/sem1.2_aokvqa_val.json')

    # the two results to compare
    results1 = load_json('../result_files/aokvqa/captions_aokvqa_val2017.json')
    results2 = load_json('../result_files/aokvqa/sem11_aokvqa_val2017.json')


    ans_list = annotations
    q_list = questions


    # get all difference between results
    diffs = []
    total = len(results1)
    for i in range(len(results1)):
        if results1[i]['answer'] != results2[i]['answer']:
            diffs.append(i)

    # random 50 indices
    seed = 42
    random.seed(seed)
    for rand in range(100):
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
        print(image_id, q_id)
        res['expansion'] = expansion[imageid_to_path(image_id)][str(q_id)]

        # image_id_to_path
        # for k in range(0, 12 - len(image_id)):
        #     image_id = '0' + image_id
        img_path = (image_id)
        caption = captions[imageid_to_path(image_id)]
        # print(caption)

        res['image_path'] = imageid_to_path(img_path)
        res['question'] = ques['question']
        res['caption'] = caption
        res['rationale'] = ques['rationales']

        # first and second answer
        res['answer_1'] = results1[rand_idx]['answer']
        res['answer_2'] = results2[rand_idx]['answer']

        answers = [a for a in ans['direct_answers']]
        acc_res1 = get_count(res['answer_1'], answers)
        acc_res2 = get_count(res['answer_2'], answers)
        res['possible_answers'] = answers
        print(res['possible_answers'])

        if acc_res1 > acc_res2:
            res['state'] = 'bad'
        elif acc_res2 > acc_res1:
            res['state'] = 'good'
        elif acc_res2 == 0 and acc_res1 == 0:
            res['state'] = 'neutral'
        else:
            res['state'] = 'both-good'

        # print(res)
        if res['state'] == 'neutral':
            # norms = grad_norms_dict[int(q_id)]

            exp_list = res['expansion'].split(".")[:5]
            print(exp_list)
            l = exp_list
            sorted_expansions = l #sorted(l, key=lambda x:x[0], reverse=True)
            image_path = f"{images_path}" + res['image_path']
            title = caption
            print(title)

            text = res['question'] + '\n Context:' + str([x for x in sorted_expansions]) + '\n\n base Answer: ' + res[
                'answer_1'] + '\nimproved Answer: ' + res['answer_2'] + '\n\n GT Answers: ' + ", ".join(
                res["possible_answers"]) + "\n" + ", ".join(res["rationale"])
            print(text)
            # print("gpt3", gpt3[str(res['question_id'])])

            # Save path - change this
            if not os.path.exists(res['state']):
                os.mkdir(res['state'])

            save_name = res['state'] + "/"+ str(res['question_id']) + '_'+ str(seed)
            show_image(image_path, text, title, save_name)

    print(len(diffs))
    print(total)
