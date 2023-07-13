
import csv

def read_data(input_file):
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

def gen_prompt(sample):
    task_prompt = "You need to annotate users' level of depression based on their posts. There are three labels: Not depressed, Moderate, and Severe.\n"
    rule_prompt = "### Here are annotating rules: \n" \
                  "# The postings data will be annotated as 'Not Depressed', if the postings data reflect one of the following mannerisms:\n" \
                  "(1) Only one or two lines about irrelevant topics;\n" \
                  "(2) Momentary feelings of present situation;\n" \
                  "(3) Asking questions about any or medication;\n" \
                  "(4) Asking/seeking help for friend’s difficulties.\n" \
                  "# The postings data will be annotated as 'Moderate', if the postings falls under these conditions:\n" \
                  "(1) Reflecting change in feelings (feeling low for some time and feeling better for some time);\n" \
                  "(2) Not feeling completely immersed in any situations; Showing hope for life.\n" \
                  "# The postings data will be annotated as 'Severe', if the postings have one of the following scenarios:\n" \
                  "(1) Expressing more than one disorder conditions;\n" \
                  "(2) Explaining about history of suicide attempts.\n"
    content_prompt = "Now please annotate the given Posts Content by following the above rules:\n" \
                     "Posts Content: {} -- Annotation: ".format(' '.join(sample))
    demo_prompt = "### Here are some examples:\n" \
                  "Posts Content: If I disappeared today, would it really matter? I’m just too tired to go on, but at the same time I’m too tired to end it. -- Annotation: Moderate \n" \
                  "Posts Content: Does anyone else get irritated easily because of depression and gets angry at people even tho you didn’t mean it? -- Annotation: Not depressed \n" \
                  "Posts Content: If my god is truly merciful, then let him kill me so that my family won’t have to deal with my suicide. Someone please kill me. -- Annotation: Severe \n"
    # if ZeroShot:
    #     return task_prompt + rule_prompt + content_prompt
    # else:
    return task_prompt + rule_prompt + demo_prompt + content_prompt

