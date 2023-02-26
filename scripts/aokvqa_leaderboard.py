from utils import load_json, save_json

answer_file = load_json('../aokvqa/captions_aokvqa_test2017.json')
#{ "question_id": { "multiple_choice": <predicted multiple choice answer (string))>, "direct_answer": <predicted open-ended answer (string)> }, }
out = {}
for original_dict in answer_file:

    out[original_dict["question_id"]] = {"multiple_choice": "", "direct_answer": original_dict["answer"]}


save_json("../aokvqa/captions_aokvqa_test2017_leaderboard.json", out)
