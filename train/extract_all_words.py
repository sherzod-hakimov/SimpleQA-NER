from string import punctuation
from gensim.utils import deaccent

def remove_accents(input):
    output = deaccent(input)
    return output

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


def extract_words():
    dataset_names = ["train", "test", "valid"]

    words = set()
    for d in dataset_names:
        dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            for i, line in enumerate(content):

                data = line.split("\t")
                text = data[3].replace("\n", "")
                tokens = text.split(" ")

                for t in tokens:
                    t = remove_accents(t)
                    t = strip_punctuation(t)
                    t = t.lower()
                    words.add(t)

    f = open('../data/words.txt', 'w')
    for w in words:
        f.write(w+"\n")
    f.close()
    print("Extracted: "+str(len(words)) +" words")