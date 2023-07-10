from fire import Fire
from argparse import Namespace
import os
from modeling import select_model, EvalModel
import numpy as np
import csv
from tqdm import tqdm
import pandas as pd
import sacrebleu
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_tsv(input_file):
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

# def read_csv_reframing(input_file):
#     data = pd.read_csv(input_file)
#     situation = list(data['situation'])
#     thought = list(data['thought'])
#     reframe = list(data['reframe'])
#     data_dic = []
#     sample = {}
#     for i in range(len(data)):
#         if sample.get('situation') == None:
#             sample['situation'] = situation[i]
#             sample['thought'] = thought[i]
#             sample['reframe1'] = reframe[i]
#         else:
#             assert situation[i] == sample['situation']
#             sample['reframe2'] = reframe[i]
#             data_dic.append(sample)
#             sample = {}
#     return data_dic

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

def read_csv_reframing(input_file):
    data = pd.read_csv(input_file)
    data_dic = []
    sample = {}
    for i in range(len(data)):
        sample['situation'] = data['situation'].iloc[i]
        sample['thought'] = data['thought'].iloc[i]
        sample['reframe'] = data['reframe'].iloc[i]
        sample['thinking_traps'] = data['thinking_traps_addressed'].iloc[i]
        data_dic.append(sample)
        sample = {}
    return data_dic
#
def gen_prompt_reframing(sample):
    task_prompt = "You need to conduct a Cognitive Reframing task, which involves replacing a negative thought with a more hopeful \"reframed thought\" that offers an alternative perspective on one's situation.\n"
    rule_prompt = "The inputs for this task are: The given Situation; The Negative Thought; The Thinking Traps that need to be addressed. The output of this task is the Reframed Thought. \n"
    demo_prompt = "For Example: \n" \
                  "Situation: A Roomate of mine stole my comptuer.\n" \
                  "Negative Thought: Someone I trusted stole something valuable of mine, I was extremely angry and wanted justice.\n" \
                  "Thinking Traps: emotional reasoning\n" \
                  "Reframed Thought: My roommate stole something of mine, and I will focus on actionable solutions to address this.\n" \
                  "Let's begin!\n"
    content_prompt = "Situation: {}\nNegative Thought:{}\nThinking Traps:{}\nReframed Thought:".format(sample['situation'], sample['thought'], sample['thinking_traps'])

    return task_prompt + rule_prompt + demo_prompt + content_prompt
#
def evaluate_reframing(model: EvalModel, test_data):
    refs = []
    answer = []

    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = gen_prompt_reframing(sample)

        ref = sample['reframe']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.split('\n')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        # print(dict(prompt=prompt, refs=ref, pred=pred))

    res = sacrebleu.corpus_bleu(answer, refs)

    # all_probs = np.array(all_probs)
    print("BLEU {:.4f} - {}".format(res.score, 'reframing'))

    return refs, answer, res.score

def read_csv_empathetic(input_file):
    data = open(input_file).readlines()
    data_dic = []
    cur_id = ''
    for i in range(1, len(data)):
        parts = data[i].strip().split(",")
        if parts[0] != cur_id: # initial
            cur_id = parts[0]
            sample = {}
            sample['conv_id'] = parts[0]
            sample['dialogue'] = ['User: ' + parts[5].replace("_comma_", ",")]
        else:
            if len(sample['dialogue']) % 2 == 1:
                tmp_s = copy.deepcopy(sample)
                sample['ref'] = parts[5].replace("_comma_", ",")
                data_dic.append(sample) # add sample
                sample = tmp_s
                sample['dialogue'].append('Assistant: ' + parts[5].replace("_comma_", ","))
            else:
                sample['dialogue'].append('User: ' + parts[5].replace("_comma_", ","))
    return data_dic
#
def gen_prompt_empathetic(sample):
    task_prompt = "As an empathetic AI assistant, you are able to recognize the emotions of your conversation partner and respond accordingly. Your task is to generate empathic responses during your conversations with users.\n"
    demo_prompt = "For Example:\n" \
                  "User: I have never cheated on my wife.\n" \
                  "Assistant: And thats something you should never do, good on you.\n" \
                  "Let's begin!\n"
    content_prompt = "{}\nAssistant: ".format('\n'.join(sample['dialogue']))
    return task_prompt + demo_prompt + content_prompt
#
def evaluate_empathetic(model: EvalModel, test_data):
    refs = []
    answer = []

    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = gen_prompt_empathetic(sample)

        ref = sample['ref']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.split('\n')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        # print(dict(prompt=prompt, refs=ref, pred=pred))

    res = sacrebleu.corpus_bleu(answer, refs)

    # all_probs = np.array(all_probs)
    print("BLEU {:.4f} - {}".format(res.score, 'empathetic'))

    return refs, answer, res.score

def main(task: str = "depression", ZeroShot: bool = True, **kwargs):
    args = Namespace(**locals())
    print(locals())

    all_results = []
    if task == 'depression':
        data_dir = '../data/depression'
        data = read_tsv(data_dir+'/test.tsv')
        model = select_model(max_input_length=2048, max_output_length=8, **kwargs)
        cors, acc, probs = evaluate_depression(model, data, ZeroShot)
        return acc
    if task == 'reframing':
        data_dir = '../data/reframing'
        data = read_csv_reframing(data_dir + '/reframing_dataset.csv')
        model = select_model(max_input_length=2048, max_output_length=60, **kwargs)
        refs, answer, res = evaluate_reframing(model, data)
        return res
    if task == 'empathetic':
        data_dir = '../data/empathetic'
        data = read_csv_empathetic(data_dir + '/test.csv')
        model = select_model(max_input_length=2048, max_output_length=60, **kwargs)
        refs, answer, res = evaluate_empathetic(model, data)
        return res
    # print(result)

#
# """
# python main.py psy --model_name seq_to_seq --model_path google/flan-t5-xl
# {'score': 0.5632458233890215}

#
if __name__ == "__main__":
    Fire()
