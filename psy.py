import random
random.seed(1)
from fire import Fire
from argparse import Namespace
import os
from modeling import select_model, EvalModel
import numpy as np
from tqdm import tqdm
import json
import sacrebleu
from sklearn.metrics import f1_score, accuracy_score
from psy_eval import empathetic, depression, reframing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def acc_f1(preds, annos):
    acc = accuracy_score(y_true = annos, y_pred = preds)
    f1 = f1_score(y_true = annos, y_pred = preds, average='macro')
    return acc, f1

def evaluate_depression(model: EvalModel, test_data, save_path):
    cors = []
    preds, annos = [], []
    results = []
    label_map = {'not depression': 0, 'moderate': 1, 'severe': 2}
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = depression.gen_prompt(sample['context'])
        post_nums = len(sample['context'])

        while not model.check_valid_length(prompt) and post_nums > 0:
            post_nums -= 1
            prompt = depression.gen_prompt(sample['context'][:post_nums])

        label = sample['label']
        pred = model.run(prompt)
        pred = pred.strip().lower()
        # probs = [0 for _ in get_choices()]
        annos.append(label_map[label])
        if 'not depress' in pred:
            preds.append(0)
        elif 'moderate' in pred:
            preds.append(1)
        elif 'severe' in pred:
            preds.append(2)
        else:
            preds.append(random.choice([0,1,2]))
        if label == 'not depression':
            cor = 'not depress' in pred
        else:
            cor = label in pred
        cors.append(cor)
        # print(dict(label=label, pred=pred))
        # print(dict(prompt=prompt, label=label, pred=pred))
        results.append(dict(sample=sample['context'], prompt=prompt, label=label, pred=pred))

    acc = np.mean(cors)
    # print(preds, annos)
    acc2, f1 = acc_f1(preds, annos)

    print(dict(acc1 = acc, acc2 = acc2, macro_f1 = f1))
    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return cors, acc, preds

def evaluate_reframing(model: EvalModel, test_data, save_path):
    refs = []
    answer = []
    results = []
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = reframing.gen_prompt(sample)

        ref = sample['reframe']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.split('\n')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        # print(dict(prompt=prompt, refs=ref, pred=pred))
        sample['prompt'] = prompt
        sample['pred'] = pred
        results.append(sample)

    res = sacrebleu.corpus_bleu(answer, refs)

    print("BLEU {:.4f} - {}".format(res.score, 'reframing'))

    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return refs, answer, res.score

def evaluate_empathetic(model: EvalModel, test_data, save_path):
    refs = []
    answer = []
    results = []
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = empathetic.gen_prompt(sample)

        ref = sample['ref']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.split('\n')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        # print(dict(prompt=prompt, refs=ref, pred=pred))
        sample['prompt'] = prompt
        sample['pred'] = pred
        results.append(sample)

    res = sacrebleu.corpus_bleu(answer, refs)

    # all_probs = np.array(all_probs)
    print("BLEU {:.4f} - {}".format(res.score, 'empathetic'))
    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return refs, answer, res.score

def main(task: str = "depression", save_path: str = './psy_eval/results/depression.json', **kwargs):
    # save_path += task + '.json'
    args = Namespace(**locals())
    print(locals())

    all_results = []
    if task == 'depression':
        data_dir = '../data/depression'
        data = depression.read_data(data_dir+'/test.tsv')
        model = select_model(max_input_length=2048, max_output_length=8, **kwargs)
        cors, acc, probs = evaluate_depression(model, data, save_path)
        return acc
    if task == 'reframing':
        data_dir = '../data/reframing'
        data = reframing.read_data(data_dir + '/reframing_dataset.csv')
        model = select_model(max_input_length=2048, max_output_length=60, **kwargs)
        refs, answer, res = evaluate_reframing(model, data, save_path)
        return res
    if task == 'empathetic':
        data_dir = '../data/empathetic'
        data = empathetic.read_data(data_dir + '/test.csv')
        model = select_model(max_input_length=2048, max_output_length=60, **kwargs)
        refs, answer, res = evaluate_empathetic(model, data, save_path)
        return res
    # print(result)

#
# """
# python main.py psy --model_name seq_to_seq --model_path google/flan-t5-xl
# {'score': 0.5632458233890215}

#
if __name__ == "__main__":
    Fire()
