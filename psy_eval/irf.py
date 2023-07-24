
import pandas as pd

def read_data(input_file):
    """Reads a tab separated value file."""
    data = pd.read_csv(input_file)
    data_dic = []
    sample = {}
    for i in range(len(data)):
        sample['context'] = data['text'].iloc[i]
        sample['TBE'] = data['belong'].iloc[i]
        sample['PBU'] = data['burden'].iloc[i]
        data_dic.append(sample)
        sample = {}
    return data_dic

def gen_prompt_TBE(sample):
    task_prompt = "You need to detect whether a person is having Thwarted Belongingness (TBE) or not based on their reddit posts. " \
                  "TBE is a psychologically-painful mental state that results from inadequacy of connectedness. " \
                  "It contains detailed set of instructions to mark latent feeling of disconnectedness, missing someone, major event " \
                  "such as death, or being ignored/ostracized/alienated, as TBE. There are two labels: 0 or 1, " \
                  "where 0 indicates No Thwarted Belongingness and 1 indicates Thwarted Belongingness present.\n"
    rule_prompt = "### Here are some additional rules: \n" \
                  "(1) TBE in the past is also be marked as presence of TBE;\n" \
                  "(2) Feelings of void/missing/regrets/or even mentioning such events with negative words should be marked as presence of TBE;\n" \
                  "(3) Anything associated with fights/quarrels/general stories should be marked with absence of TBE;\n"
    demo_prompt = "### Here are some examples:\n" \
                  "Posts Content: I was so upset being lonely before Christmas and today I am celebrating New Year with friends.\n" \
                  "Answer: 1\n" \
                  "Posts Content: But I just miss her SO. much. It’s like she set the bar so high that all I can do is just stare at it.\n" \
                  "Answer: 1 \n" \
                  "Posts Content: My husband and I just had a huge argument and he stormed out. I should be crying or stopping him or something. But I decided to take a handful of benzos instead.\n" \
                  "Answer: 0 \n"
    content_prompt = "### \n" \
                     "Posts Content: {}\n" \
                    "Answer: ".format(sample['context'])
    return task_prompt + rule_prompt + demo_prompt + content_prompt

def gen_prompt_PBU(sample):
    task_prompt = "You need to detect whether a person is having Perceived Burdensomeness (PBU) or not based on their reddit posts. " \
                  "PBU is a mental state characterized by making fully conscious perception that others would “be better off if I were gone,” which manifests when the need for social competence." \
                  "There are two labels: 0 or 1, " \
                  "where 0 indicates No Perceived Burdensomeness and 1 indicates Perceived Burdensomeness present.\n"
    # rule_prompt = "### Here is an additional rules: \n" \
    #               "(1) PBU in the past is also be marked as presence of PBU;\n"
    demo_prompt = "### Here are some examples:\n" \
                  "Posts Content: I don’t have any friends, I’ve never been in a proper, loving relationship and I’m a socially awkward loser. " \
                  "Other people see me as a burden, people hate talking to me, and I’m tired of continuing on with this.\n" \
                  "Answer: 1\n" \
                  "Posts Content: I only take Lexapro. I was watching some videos on these guy that call themselves \"Preppers\" and " \
                  "they prep for the end of the world. They say that people on any types of drugs will become unstable and focused on getting their fix or whatever. Is that us?.\n" \
                  "Answer: 0 \n"
    content_prompt = "### \n" \
                     "Posts Content: {}\n" \
                    "Answer: ".format(sample['context'])
    return task_prompt + demo_prompt + content_prompt

