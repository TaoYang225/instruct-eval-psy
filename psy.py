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
from psy_eval import empathetic, depression, reframing, mental, stress, irf
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def acc_f1(preds, annos, average='macro'):
    acc = accuracy_score(y_true = annos, y_pred = preds)
    f1 = f1_score(y_true = annos, y_pred = preds, average=average)
    return acc, f1

def evaluate_depression(model: EvalModel, test_data, save_path):
    cors = []
    preds, annos = [], []
    results = []
    label_map = {'not depression': 0, 'moderate': 1, 'severe': 2}
    start = True
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = depression.gen_prompt(sample['context'])
        post_nums = len(sample['context'])

        while not model.check_valid_length(prompt) and post_nums > 0:
            post_nums -= 1
            prompt = depression.gen_prompt(sample['context'][:post_nums])

        label = sample['label']
        pred = model.run(prompt)
        pred = pred.strip().lower().split('\n')[0]
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
        if start:
            print(dict(prompt=prompt, label=label, pred=pred))
            start = False
        results.append(dict(sample=sample['context'], prompt=prompt, label=label, pred=pred))

    acc = np.mean(cors)
    # print(preds, annos)
    acc2, f1 = acc_f1(preds, annos)

    print(dict(acc1 = acc, acc2 = acc2, macro_f1 = f1))
    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return cors, acc, preds

def evaluate_stress(model: EvalModel, test_data, save_path):
    cors = []
    preds, annos = [], []
    results = []
    # label_map = {'0': 0, '1': 1}
    start = True
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = stress.gen_prompt(sample['text'])

        label = sample['label']
        pred = model.run(prompt)
        pred = pred.strip().lower()
        # probs = [0 for _ in get_choices()]
        annos.append(label)
        if '0' in pred:
            preds.append(0)
        elif '1' in pred:
            preds.append(1)
        else:
            preds.append(random.choice([0,1]))

        cor = str(label) in pred
        cors.append(cor)
        # print(dict(label=label, pred=pred))
        if start:
            print(dict(prompt=prompt, label=label, pred=pred))
            start = False
        results.append(dict(sample=sample['text'], prompt=prompt, label=str(label), pred=pred))

    acc = np.mean(cors)
    # print(preds, annos)
    acc2, f1 = acc_f1(preds, annos, average='binary')

    print(dict(acc1 = acc, acc2 = acc2, macro_f1 = f1))
    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return cors, acc, preds

def evaluate_irf(model: EvalModel, test_data, save_path):
    cors_tbe, cors_pbu = [], []
    preds_tbe, annos_tbe, preds_pbu, annos_pbu = [], [], [], []
    results = []
    # label_map = {'0': 0, '1': 1}
    start = True
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt_tbe = irf.gen_prompt_TBE(sample)
        prompt_pbu = irf.gen_prompt_PBU(sample)

        label_tbe = sample['TBE']
        label_pbu = sample['PBU']
        pred_tbe = model.run(prompt_tbe)
        pred_pbu = model.run(prompt_pbu)

        pred_tbe = pred_tbe.strip().lower()
        pred_pbu = pred_pbu.strip().lower()
        # probs = [0 for _ in get_choices()]
        annos_tbe.append(label_tbe)
        annos_pbu.append(label_pbu)
        if '0' in pred_tbe:
            preds_tbe.append(0)
        elif '1' in pred_tbe:
            preds_tbe.append(1)
        else:
            preds_tbe.append(random.choice([0,1]))

        if '0' in pred_pbu:
            preds_pbu.append(0)
        elif '1' in pred_pbu:
            preds_pbu.append(1)
        else:
            preds_pbu.append(random.choice([0,1]))

        cor_tbe = str(label_tbe) in pred_tbe
        cors_tbe.append(cor_tbe)
        cor_pbu = str(label_pbu) in pred_pbu
        cors_pbu.append(cor_pbu)
        # print(dict(label=label, pred=pred))
        if start:
            print(dict(prompt_tbe=prompt_tbe, prompt_pbu=prompt_pbu, label_tbe=label_tbe, pred_tbe=pred_tbe, label_pbu=label_pbu, pred_pbu=pred_pbu))
            start = False
        results.append(dict(sample=sample['context'], prompt_tbe=prompt_tbe, prompt_pbu=prompt_pbu, label_tbe=str(label_tbe), pred_tbe=pred_tbe, label_pbu=str(label_pbu), pred_pbu=pred_pbu))

    acc_tbe = np.mean(cors_tbe)
    acc_pbu = np.mean(cors_pbu)
    # print(preds, annos)
    acc2_tbe, f1_tbe = acc_f1(preds_tbe, annos_tbe, average='binary')
    acc2_pbu, f1_pbu = acc_f1(preds_pbu, annos_pbu, average='binary')

    print(dict(acc1_tbe = acc_tbe, acc2_tbe = acc2_tbe, macro_f1 = f1_tbe))
    print(dict(acc1_pbu = acc_pbu, acc2_pbu = acc2_pbu, macro_f1 = f1_pbu))
    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return cors_pbu, acc_pbu, preds_pbu

def evaluate_reframing(model: EvalModel, test_data, save_path):
    refs = []
    answer = []
    results = []
    start = True
    for sample in tqdm(test_data[1:]):
        # get prompt and make sure it fits
        prompt = reframing.gen_prompt(sample)

        ref = sample['reframe']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.split('\n')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        if start:
            print(dict(prompt=prompt, refs=ref, pred=pred))
            start = False
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
    start = False
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = empathetic.gen_prompt(sample)

        ref = sample['ref']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.split('\n')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        if start:
            print(dict(prompt=prompt, refs=ref, pred=pred))
            start = False
        sample['prompt'] = prompt
        sample['pred'] = pred
        results.append(sample)

    res = sacrebleu.corpus_bleu(answer, refs)

    # all_probs = np.array(all_probs)
    print("BLEU {:.4f} - {}".format(res.score, 'empathetic'))
    with open(save_path, 'w') as fp:
        json.dump(results, fp)
    return refs, answer, res.score

def evaluate_mental(model: EvalModel, test_data, save_path):
    refs = []
    answer = []
    results = []
    start = True
    for sample in tqdm(test_data):
        # get prompt and make sure it fits
        prompt = mental.gen_prompt(sample)

        ref = sample['answers']
        refs.append(ref)
        pred = model.run(prompt)
        pred = pred.strip().split('USER:')[0]
        answer.append(pred)
        # print(dict(label=label, pred=pred))
        if start:
            print(dict(prompt=prompt, refs=ref, pred=pred))
            start = False
        sample['prompt'] = prompt
        sample['pred'] = pred
        results.append(sample)

    res = sacrebleu.corpus_bleu(answer, refs)

    print("BLEU {:.4f} - {}".format(res.score, 'Mental Health FAQ'))

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
    if task == 'stress':
        data_dir = '../data/stress'
        data = stress.read_data(data_dir+'/test.pkl')
        model = select_model(max_input_length=2048, max_output_length=5, **kwargs)
        cors, acc, probs = evaluate_stress(model, data, save_path)
        return acc
    if task == 'IRF':
        data_dir = '../data/IRF'
        data = irf.read_data(data_dir+'/test_data.csv')
        model = select_model(max_input_length=2048, max_output_length=5, **kwargs)
        cors, acc, probs = evaluate_irf(model, data, save_path)
        return acc
    if task == 'reframing':
        data_dir = '../data/reframing'
        data = reframing.read_data(data_dir + '/reframing_dataset.csv')
        model = select_model(max_input_length=2048, max_output_length=60, **kwargs)
        refs, answer, res = evaluate_reframing(model, data, save_path)
        return res
    if task == 'mental':
        data_dir = '../data/mh_faq'
        data = mental.read_data(data_dir + '/Mental_Health_FAQ.csv')
        model = select_model(max_input_length=200, max_output_length=500, **kwargs)
        refs, answer, res = evaluate_mental(model, data, save_path)
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
