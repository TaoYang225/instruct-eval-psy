
import pandas as pd

def read_data(input_file):
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
def gen_prompt(sample):
    task_prompt = "You need to conduct a Cognitive Reframing task, which involves replacing a negative thought with a more hopeful \"reframed thought\" that offers an alternative perspective on one's situation.\n"
    rule_prompt = "### The inputs for this task are:\n" \
                  "(1) The given Situation;\n" \
                  "(2) The Negative Thought;\n" \
                  "(3) The Thinking Traps that need to be addressed.\n" \
                  "### The output of this task is the Reframed Thought.\n"
    demo_prompt = "### Here is an example:\n" \
                  "Situation: A Roomate of mine stole my comptuer.\n" \
                  "Negative Thought: Someone I trusted stole something valuable of mine, I was extremely angry and wanted justice.\n" \
                  "Thinking Traps: emotional reasoning\n" \
                  "Reframed Thought: My roommate stole something of mine, and I will focus on actionable solutions to address this.\n"
    content_prompt = "### Now please reframe the negative thought:\n" \
                     "Situation: {}\n" \
                     "Negative Thought:{}\n" \
                     "Thinking Traps:{}\n" \
                     "Reframed Thought:".format(sample['situation'], sample['thought'], sample['thinking_traps'])
    return task_prompt + rule_prompt + demo_prompt + content_prompt

