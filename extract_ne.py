import nltk
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import json
import string
import operator
import unidecode
from difflib import SequenceMatcher
from string import punctuation
from gensim.utils import deaccent
import re

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def is_invalid_span(new_start_index, new_end_index, candidates:list, stemmer):

    for start_index, end_index, ngram, uri, freq in candidates:
        if start_index <= new_start_index and end_index >= new_end_index:

            ## if the bigger ngram start with "a", "the", "an", "of", "on", "at", "by"
            ### then allow the smaller ngram to be a candidate
            normalized_ngram = normalize_string(ngram, stemmer)
            prefixes = ["a", "the", "an", "of", "on", "at", "by", "in"]
            for p in prefixes:
                if normalized_ngram.startswith(p):
                    return False
            return True
    return False

def is_valid_ngram(stopwords, ngram):
    for token in ngram:
        normalized_token = strip_punctuation(token).lower()
        if normalized_token not in stopwords:
            return True
    return False

def combine_tokens(ngram_tokens):
    ngram = ""
    for i, token in enumerate(ngram_tokens):
        if i == len(ngram_tokens)-1:
            ngram += token
        else:
            ngram += token +" "
    return ngram

def normalize_string(input, stemmer):

    input  = input.replace("\n","").replace("'s", "")

    ngram = remove_accents(input)
    ## remove everything except alphanumeric
    ngram = re.sub('[^0-9a-zA-Z]+', '', ngram).strip().lower()
    ngram = strip_punctuation(ngram)

    return ngram

def span(ngram_tokens, tokens):

    start_index = tokens.index(ngram_tokens[0])
    end_index = tokens.index(ngram_tokens[-1])+1
    return start_index, end_index

def remove_accents(input):
    # output = unidecode.unidecode(input)
    output = deaccent(input)
    return output

def extract_subjects(sentence, dict, max_ngram_size, stopwords,exclude_small_ngrams, exclude_stop_words, stemmer):
    candidates = list()
    tokens = sentence.split(" ")
    ngram_size = max_ngram_size
    while ngram_size > 0:
        extracted_ngrams = ngrams(tokens, ngram_size)
        for ngram_tokens in extracted_ngrams:

            ##check if the span for this ngram has been covered before
            ## skip ngrams that are part of bigger ngram
            start_index, end_index = span(ngram_tokens, tokens)

            if exclude_small_ngrams:
                if is_invalid_span(start_index, end_index, candidates, stemmer):
                    continue

            ##skip ngrams that consist of only stopwords
            if exclude_stop_words:
                if not is_valid_ngram(stopwords, ngram_tokens):
                    continue

            ngram = combine_tokens(ngram_tokens)
            normalized_ngram = normalize_string(ngram, stemmer)
            if len(normalized_ngram) < 3:
                continue

            # print("\t"+ngram + " -> "+ normalized_ngram)

            if normalized_ngram in dict.keys():

                matches = dict[normalized_ngram]
                for uri in matches.keys():
                    freq = matches[uri]
                    candidates.append([start_index, end_index, ngram, uri, freq])
        ##decrease the ngram size
        ngram_size = ngram_size - 1
    return candidates


def load_index(file_path, stopwords,stemmer):
    dict = {}
    inverted_dict={}

    with open(file_path, encoding="utf-8") as f:
        content = f.readlines()
        for entry in content:

            entries = entry.split("\t")
            if len(entries) < 2:
                continue

            uri = entries[0].replace("www.freebase.com/m/", "m.")
            mention = entries[1]

            if len(mention) > 50:
                continue

            if len(mention) < 3:
                continue
            ## A / B -> take the B part
            if " / " in mention:
                mention = mention.split(" / ")[1]
            if ", " in mention:
                mention = mention.split(", ")[0]

            ### NORMALIZATION
            mention = normalize_string(mention, stemmer)

            if mention in stopwords:
                continue

            if mention in dict.keys():
                uris = dict[mention]
                # increment
                prev_count = 0
                if uri in uris.keys():
                    prev_count = uris[uri]
                uris[uri] = prev_count + 1
                dict[mention] = uris
            else:
                uris = {}
                uris[uri] = 1
                dict[mention] = uris


            ## add also to inverted_dictionary
            if uri in inverted_dict.keys():
                added_mentions = inverted_dict[uri]
                added_mentions.add(entries[1])
                inverted_dict[uri] = added_mentions
            else:
                added_mentions = set()
                added_mentions.add(entries[1])
                inverted_dict[uri] = added_mentions

    return dict, inverted_dict

def load_subject_predicates(file_path):
    dict = {}

    with open(file_path) as f:
        content = f.readlines()
        for entry in content:

            entries = entry.split("\t")
            subject = data[0].replace("www.freebase.com/m/", "m.")
            predicate = data[1].replace("www.freebase.com/", "")


            if subject in dict.keys():
                predicates = dict[subject]
                predicates.add(predicate)
                dict[subject] = predicates
            else:
                predicates = set()
                predicates.add(predicate)
                dict[subject] = predicates
    return dict


if __name__ == "__main__":
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer("english")

    ## get stop words and normalize
    initial_stopwords = nltk.corpus.stopwords.words('english')
    stopwords = list()
    for s in initial_stopwords:
        s = normalize_string(s, stemmer)
        stopwords.append(s)



    print("Loading index")
    mention_dict, inverted_mention_dict = load_index("data/surface_forms.txt", stopwords, stemmer)
    subject_predicates_dict =[]# load_subject_predicates("data/SimpleQuestions_v2/freebase-FB2M.txt")

    dataset_names = ["test"]
    max_ngram_size = 10
    exclude_small_ngrams = True
    exclude_stop_words = True


    for d in dataset_names:
        correct_count = 0
        incorrect_count = 0
        total_count = 0
        number_of_candidates = 0
        dataset_path = "data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        recall_ranges = [1, 10, 40, 50, 100, 200, 300, 500, 1000, 2000, 10000]
        recall_at_k={}

        ##initialize with 0 count
        for r in recall_ranges:
            recall_at_k[r] = 0

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            for i, line in enumerate(content):

                total_count += 1

                data = line.split("\t")
                target_subject = data[0].replace("www.freebase.com/m/", "m.")
                target_predicate = data[1].replace("www.freebase.com/", "")
                text = data[3].replace("\n","")

                subjects = extract_subjects(text, mention_dict, max_ngram_size, stopwords, exclude_small_ngrams, exclude_stop_words, stemmer)

                number_of_candidates += len(subjects)
                subjects.sort(key=lambda tup: tup[4])  # sorts in place

                is_found = False
                ### crop the list and compare recall@k
                for range in recall_ranges:
                    ##no need to crop again if it's found on prev k number
                    if is_found:
                        recall_at_k[range] = recall_at_k[range]+1
                        continue

                    top_k = min(range, len(subjects))
                    filtered_subjects = subjects[:top_k]

                    for start_index, end_index, ngram, uri, freq in filtered_subjects:
                        if uri == target_subject:
                            is_found = True
                            break

                    if is_found:
                        recall_at_k[range] = recall_at_k[range]+1


                if is_found:
                    correct_count = correct_count + 1
                else:
                    incorrect_count+=1


                    a = []
                    if target_subject in inverted_mention_dict.keys():
                        print(text)
                        print(target_subject)

                        a = inverted_mention_dict[target_subject]
                        print("Available mentions: "+ str(a))

                        found_ngrams = set()
                        for start_index, end_index, ngram, uri, freq in subjects:
                            found_ngrams.add(ngram)

                        print("Found ngrams: "+ str(found_ngrams)+"\n")

            score = correct_count / float(total_count)
            avg_number_of_candidates = number_of_candidates/float(total_count)

            print("Dataset: "+ d+ " Upper bound: " + str(score))
            print("Average #candidates: " + str(avg_number_of_candidates))

            for k in recall_ranges:
                recall_at_k_score = recall_at_k[k]/float(total_count)
                print("\tRecall@"+str(k)+" : "+str(recall_at_k_score))
