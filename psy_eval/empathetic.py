
import pandas as pd
import copy

def read_data(input_file):
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
def gen_prompt(sample):
    task_prompt = "As an empathetic AI assistant, you are able to recognize the emotions of your conversation partner and respond accordingly. Your task is to generate empathic responses during your conversations with users.\n"
    demo_prompt = "### Here is an example:\n" \
                  "User: I have never cheated on my wife.\n" \
                  "Assistant: And thats something you should never do, good on you.\n"
    content_prompt = "### Now please response the user with empathetic\n" \
                     "{}\n" \
                     "Assistant: ".format('\n'.join(sample['dialogue']))
    return task_prompt + demo_prompt + content_prompt

