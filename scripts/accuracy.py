import json

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_accuracy(results_json, i):
    annotations = _load_json('../data/coco/okvqa/mscoco_val2014_annotations.json')

    results = {}
    for r in results_json:
        results[r['question_id']] = r['answer']

    accs = []
    for annotation in annotations['annotations']:
        qid = annotation['question_id']
        answers = [a['answer'] for a in annotation['answers']]
        try:
            answer = results[qid]
            #print(answer, answers)
        except:
            print(qid)
            accs.append(0)
            continue
        human_count = answers.count(answer)
        acc = min(1.0, float(human_count) / 3)
        accs.append(acc)
        
    print(i, round(sum(accs) / len(accs), 4))

if __name__ == '__main__':

    result_base_path = 'semv2_okvqa_val2014.json'

    keys = [4, 9, 14, 19]
    for i in keys:
        results_json = _load_json('../results_okvqa/'+str(i)+"_"+result_base_path)
        get_accuracy(results_json, i)
    
                

    