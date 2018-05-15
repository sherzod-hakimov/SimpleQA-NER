from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
import json
import operator
from difflib import SequenceMatcher

def extract_subjects(sentence, dict, max_ngram):

    candidates = set()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)

    ngram_size = max_ngram
    while ngram_size > 0:
        grams = ngrams(tokens, ngram_size)
        for n in grams:
            query = ""
            for t in n:
                query+=t.lower()+""
            query = query.strip()

            if query in dict.keys():
                for uri in dict[query]:
                    candidates.add(uri)
        ##decrease the ngram size
        ngram_size=ngram_size-1
    return candidates

def load_index(file_path):
    tokenizer = RegexpTokenizer(r'\w+')

    dict = {}

    with open(file_path) as f:
        content = f.readlines()
        for entry in content:

            entries = entry.split("\t")
            if len(entries) < 2:
                continue

            uri = entries[0].replace("www.freebase.com/m/", "m.")
            mention = entries[1]

            if len(mention) > 50:
                continue

            tokens = tokenizer.tokenize(mention)

            mention = ""
            for t in tokens:
                mention += t.lower()+""
            mention = mention.strip()

            if mention in dict.keys():
                uris = dict[mention]
                uris.append(uri)

                dict[mention] = uris
            else:
                uris = list()
                uris.append(uri)
                dict[mention] = uris
    return dict

def load_degree_count(file_path):

    dict = {}
    with open(file_path) as f:
        content = f.readlines()
        for entry in content:
            entries = entry.split("\t")
            if len(entries) < 2:
                continue

            uri = entries[0]
            count = entries[1]
            dict[uri] = count
    return dict


print("Loading index")
mention_dict = load_index("data/entitySurfaceForms.txt")

print("Loading degree index")
degree_dict = load_index("data/freebase-node-degree-counts.txt")


print("Looping in sentences")

correct_count = 0
with open('data/test_all_ngrams.txt') as f:
    content = f.readlines()

    upper_bound_count = 0
    empty_prediction_count = 0
    empty_candidate_count = 0

    for i, line in enumerate(content):

        json_data = json.loads(line)
        text = json_data["text"]
        target_subject = json_data["subject"]

        uris= extract_subjects(text, mention_dict, 6)

        matches = {}
        for u in uris:
            degree_count = 0
            if u in degree_dict.keys():
                degree_dict = degree_dict[u]
            matches[u]=degree_count

        # sort predicates by similarity score
        sorted_tuples = sorted(matches.items(), key=operator.itemgetter(1), reverse=True)

        is_found =False

        top_k = min(10, len(sorted_tuples))
        for subject, degree in sorted_tuples[:top_k]:
            if subject == target_subject:
                is_found = True
                break

        if is_found:
            correct_count = correct_count+1
        else:
            print(text+" : " +target_subject + " found: "+str(len(sorted_tuples)))

    score = correct_count/float(len(content))
    print("Upper bound: "+ str(score))