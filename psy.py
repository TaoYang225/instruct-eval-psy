from fire import Fire
from argparse import Namespace
import os
from modeling import select_model, EvalModel
import numpy as np
import csv
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar="\t")
        data = []
        for lines in reader:
            if lines[0] == 'Pid':
                continue
            for line in lines:
                if line.startswith('test_pid'):
                    if line != 'test_pid_1':
                        assert len(sample) == 3
                        data.append(sample)
                    sample = {'id': line, 'context': []}
                elif line == 'moderate' or line == 'severe' or line == 'not depression':
                    sample['label'] = line
                else:
                    sample['context'].append(line)
        return data

def gen_prompt_depression(sample, ZeroShot):
    task_prompt = "You need to annotate users' level of depression based on their posts. There are three labels: Not depressed, Moderate, and Severe.\n"
    rule_prompt = "To annotate a user as 'Not Depressed', it must reflect one of the following mannerisms: " \
                  "Only one or two lines about irrelevant topics; " \
                  "Momentary feelings of present situation; " \
                  "Asking questions about any or medication; " \
                  "Asking/seeking help for friend’s difficulties. \n" \
                  "To annotate a user as 'Moderately depressed', it must fall under one of these conditions: " \
                  "Reflecting change in feelings (feeling low for some time and feeling better for some time); " \
                  "Not feeling completely immersed in any situations; Showing hope for life. \n" \
                  "To annotate a user as 'Severely depressed', it must have one of the following scenarios: " \
                  "Expressing more than one disorder condition; Explaining about history of suicide attempts.\n"
    content_prompt = "Posts Content: {} -- ".format(' '.join(sample))
    question_prompt = "Annotation: "
    demo_prompt = "Demonstrations: \n" \
                  "Posts Content: I don’t want to die I just want to stop living. : Does that even make sense? God I hate my life. -- Annotation: Moderate \n" \
                  "Posts Content: Does anyone else get irritated easily because of depression and gets angry at people even tho you didn’t mean it? -- Annotation: Not depressed \n" \
                  "Posts Content: If my god is truly merciful, then let him kill me so that my family won’t have to deal with my suicide. Someone please kill me. -- Annotation: Severe \n"
    if ZeroShot:
        return task_prompt + rule_prompt + content_prompt + question_prompt
    else:
        return task_prompt + rule_prompt + demo_prompt + content_prompt + question_prompt

def evaluate_depression(model: EvalModel, test_data, ZeroShot):
    cors = []
    answer = []

    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = gen_prompt_depression(sample['context'], ZeroShot)
        post_nums = len(sample['context'])

        while not model.check_valid_length(prompt) and post_nums > 0:
            post_nums -= 1
            prompt = gen_prompt_depression(sample['context'][:post_nums], ZeroShot)

        label = sample['label']
        pred = model.run(prompt)
        # probs = [0 for _ in get_choices()]
        if label == 'not depression':
            cor = 'not depress' in pred.strip().lower()
        else:
            cor = label in pred.strip().lower()
        cors.append(cor)
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        # print(dict(prompt=prompt, label=label, pred=pred))

    acc = np.mean(cors)
    cors = np.array(cors)

    # all_probs = np.array(all_probs)
    # print("Average accuracy {:.3f} - {}".format(acc, 'depression'))

    return cors, acc, answer

def main(task: str = "depression", ZeroShot: bool = True, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=2048, max_output_length=8, **kwargs)
    print(locals())

    all_results = []
    if task == 'depression':
        data_dir = '../data/depression'
        data = read_tsv(data_dir+'/test.tsv')
        cors, acc, probs = evaluate_depression(model, data, ZeroShot)
        return acc
    # print(result)

#
# """
# python main.py psy --model_name seq_to_seq --model_path google/flan-t5-xl
# {'score': 0.5632458233890215}

#
if __name__ == "__main__":
    Fire()
