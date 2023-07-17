
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def pre_process(input_file):
    raw_df = pd.read_csv(input_file)
    data = []
    for i in range(len(raw_df)):
        sample = {}
        sample['subreddit'] = raw_df['subreddit'].iloc[i]
        sample['post_id'] = raw_df['post_id'].iloc[i]
        sample['text'] = raw_df['text'].iloc[i]
        sample['label'] = raw_df['label'].iloc[i]
        sample['confidence'] = raw_df['confidence'].iloc[i]
        sample['social_timestamp'] = raw_df['social_timestamp'].iloc[i]
        data.append(sample)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    with open('../../instruct-eval-main/data/stress/train.pkl', 'wb') as wt:
        pickle.dump(train, wt)

    with open('../../instruct-eval-main/data/stress/test.pkl', 'wb') as wt:
        pickle.dump(test, wt)
    return train, test


def read_data(input_file):
    """Reads a pkl file."""
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
        return data

def gen_prompt(sample):
    task_prompt = "You need to detect whether a person is having psychological stress or not based on their reddit posts. There are two labels: 0 or 1, where 0 indicates no stress and 1 indicates stress.\n"
    demo_prompt = "### Here are some examples:\n" \
                  "Posts Content: I have anxiety ptsd depression and a severe eating disorder, she has severely crippling anxiety she needs meds for that we have to pay out of pocket for. " \
                  "We have until Feb on this lease and then we are done with this hell >We dont buy anything other tthan necessities but the rents so fucking high " \
                  ">We dont even have car insurance anymore bc we cant afford it >Often I have to work on an empty stomach, days at a time\n" \
                  "Answer: 1 \n" \
                  "Posts Content: She is friends with a buddy of mine, but her and I are not even facebook friends or anything, we talked five or six times in these two years and never anything big, just small talk. " \
                  "A few months after starting classes, my buddy told me that Chloe has a  crush on me, and gave me the impression that she wanted me to be aware of this, " \
                  "however, even after that, she didn’t try to talk to me or make herself noticed at all. She is a pretty girl, the artsy kind that likes to travel, read poetry and paint, coincidentally the exact opposite of my girlfriend who has very different hobbies. " \
                  "Like I said, I love Alice and I didn’t think much of it. Her avoidant behaviour hasn’t changed and I didn’t approach her either.\n" \
                  "Answer: 0 \n"
    content_prompt = "Posts Content: {}\n" \
                     "Answer: ".format(sample)

    return task_prompt + demo_prompt + content_prompt

# train, test = pre_process('../../instruct-eval-main/data/stress/Stress.csv')
# test2 = read_data('../../instruct-eval-main/data/stress/test.pkl')