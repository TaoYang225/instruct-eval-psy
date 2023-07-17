
import pandas as pd

def read_data(input_file):
    data = pd.read_csv(input_file)
    data['Questions'] = data['Questions'].str.lower()
    data.dropna(inplace=True)
    data_dic = []
    sample = {}
    for i in range(len(data)):
        sample['question'] = data['Questions'].iloc[i]
        sample['answers'] = data['Answers'].iloc[i]
        data_dic.append(sample)
        sample = {}
    return data_dic
#
def gen_prompt(sample):
    task_prompt = "A chat between a curious user and an artificial intelligence assistant. " \
                  "The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:".format(sample['question'])
    return task_prompt

#
# data = read_data('../../instruct-eval-main/data/mh_faq/Mental_Health_FAQ.csv')