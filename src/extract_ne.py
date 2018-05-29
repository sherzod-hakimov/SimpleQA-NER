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
from difflib import SequenceMatcher
import re


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def stopwords():
    initial_stopwords = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against",
        "all", "almost", "alone", "along", "already", "also", "although", "always",
        "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
        "around", "as", "at", "back", "be", "became", "because", "become",
        "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both",
        "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
        "could", "couldnt", "cry", "de", "describe", "detail", "did", "do", "does", "done",
        "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
        "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
        "find", "fire", "first", "five", "for", "former", "formerly", "forty",
        "found", "four", "from", "front", "full", "further", "get", "give", "go",
        "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
        "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
        "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
        "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
        "latterly", "least", "less", "ltd", "made", "many", "may", "me",
        "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
        "move", "much", "must", "my", "myself", "name", "namely", "neither",
        "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
        "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
        "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
        "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
        "please", "put", "rather", "re", "same", "see", "seem", "seemed",
        "seeming", "seems", "serious", "several", "she", "should", "show", "side",
        "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
        "something", "sometime", "sometimes", "somewhere", "still", "such",
        "system", "take", "ten", "than", "that", "the", "their", "them",
        "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
        "third", "this", "those", "though", "three", "through", "throughout",
        "thru", "thus", "to", "together", "too", "top", "toward", "towards",
        "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
        "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
        "whence", "whenever", "where", "whereafter", "whereas", "whereby",
        "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
        "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
        "within", "without", "would", "yet", "you", "your", "yours", "yourself",
        "yourselves"]
    return initial_stopwords


def is_invalid_span(new_start_index, new_end_index, candidates: list, stemmer):
    for start_index, end_index, ngram, uri, freq in candidates:
        if start_index <= new_start_index and end_index >= new_end_index:

            ## if the bigger ngram start with "a", "the", "an", "of", "on", "at", "by"
            ### then allow the smaller ngram to be a candidate
            prefixes = ["a", "the", "an", "of", "on", "at", "by", "in"]
            for p in prefixes:
                if p == ngram.lower().split(" ")[0]:
                    return False
                # if normalized_ngram.startswith(p):
                #     return False
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
        if i == len(ngram_tokens) - 1:
            ngram += token
        else:
            ngram += token + " "
    return ngram


def normalize_string(input):
    input = input.replace("\n", "").replace("'s", "")

    # if len(input) > 3:
    #     if input[-1] == "s":
    #         input = stemmer.stem(input)

    ngram = remove_accents(input)
    ## remove everything except alphanumeric
    ngram = re.sub('[^0-9a-zA-Z]+', '', ngram).strip().lower()
    ngram = strip_punctuation(ngram)




    return ngram


def span(ngram_tokens, tokens):
    start_index = tokens.index(ngram_tokens[0])
    end_index = tokens.index(ngram_tokens[-1]) + 1
    return start_index, end_index


def remove_accents(input):
    # output = unidecode.unidecode(input)
    output = deaccent(input)
    return output

def extract_candidates(ngram, dict, partial_match=False):
    normalized_ngram = normalize_string(ngram)

    candidates = list()

    if partial_match:
        for key in dict.keys:
            similarity_score = SequenceMatcher(None, key, normalized_ngram).ratio()

            if similarity_score >=0.8:
                matches = dict[key]

                for uri in matches.keys():
                    freq = matches[uri]
                    candidates.append([start_index, end_index, ngram, uri, freq])
    else:
        if normalized_ngram in dict.keys():
            matches = dict[normalized_ngram]
            for uri in matches.keys():
                freq = matches[uri]
                candidates.append([start_index, end_index, ngram, uri, freq])

    return candidates

def extract_subjects(sentence, dict, max_ngram_size, stopwords, exclude_small_ngrams, exclude_stop_words, stemmer):
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
            normalized_ngram = normalize_string(ngram)
            if len(normalized_ngram) < 3:
                continue

            if normalized_ngram in dict.keys():

                matches = dict[normalized_ngram]

                for uri in matches.keys():
                    freq = matches[uri]
                    candidates.append([start_index, end_index, ngram, uri, freq])
        ##decrease the ngram size
        ngram_size = ngram_size - 1
    return candidates


def load_index(file_path):
    dict = {}

    with open(file_path, encoding="utf-8") as f:
        content = f.readlines()
        for entry in content:

            entries = entry.split("\t")

            mention  = entries[0]
            uri = entries[1]
            freq = int(entries[2])

            if mention in dict.keys():
                uris = dict[mention]
                uris[uri] = freq
                dict[mention] = uris
            else:
                uris = {}
                uris[uri] = freq
                dict[mention] = uris
    return dict


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


def load_subject_triple_counts(file_path):
    dict = {}

    with open(file_path) as f:
        content = f.readlines()
        for entry in content:
            entries = entry.split("\t")
            if len(entries) != 2:
                continue

            subject = entries[0]
            count = int(entries[1].replace("\n", ""))

            dict[subject] = count

    return dict


if __name__ == "__main__":
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer("english")

    ## get stop words and normalize
    initial_stopwords = stopwords()
    stopwords = list()
    for s in initial_stopwords:
        s = normalize_string(s)
        stopwords.append(s)

    print("Loading index")
    mention_dict = load_index("../data/surface_forms_new.txt")
    subject_predicates_dict = []  # load_subject_predicates("data/SimpleQuestions_v2/freebase-FB2M.txt")
    subject_triple_counts = []#load_subject_triple_counts("data/subject_triple_counts.txt")

    dataset_names = ["test"]
    max_ngram_size = 10
    exclude_small_ngrams = True
    exclude_stop_words = True

    for d in dataset_names:
        correct_count = 0
        incorrect_count = 0
        total_count = 0
        number_of_candidates = 0

        dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        recall_ranges = [1, 5, 10, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 10000]
        recall_at_k = {}

        ##initialize with 0 count
        for r in recall_ranges:
            recall_at_k[r] = 0

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            for i, line in enumerate(content):

                total_count += 1

                data = line.split("\t")
                target_subject = data[0].replace("www.freebase.com/m/", "m.")
                target_predicate = data[1].replace("www.freebase.com/", "").replace("/",".")
                text = data[3].replace("\n", "")

                candidates = extract_subjects(text, mention_dict, max_ngram_size, stopwords, exclude_small_ngrams,
                                            exclude_stop_words, stemmer)

                subjects = list()

                valid_ngram = ""
                for start_index, end_index, ngram, uri, freq in candidates:
                    if uri == target_subject:
                        valid_ngram = ngram
                        break

                for start_index, end_index, ngram, uri, freq in candidates:
                    if valid_ngram == ngram:
                        subjects.append([start_index, end_index, ngram, uri, freq])


                number_of_candidates += len(subjects)
                subjects.sort(key=lambda tup: tup[4])  # sorts in place

                is_found = False
                ### crop the list and compare recall@k
                for range in recall_ranges:
                    ##no need to crop again if it's found on prev k number
                    if is_found:
                        recall_at_k[range] = recall_at_k[range] + 1
                        continue

                    top_k = min(range, len(subjects))
                    filtered_subjects = subjects[:top_k]

                    for start_index, end_index, ngram, uri, freq in filtered_subjects:
                        if uri == target_subject:
                            is_found = True
                            break

                    if is_found:
                        recall_at_k[range] = recall_at_k[range] + 1

                if is_found:
                    correct_count = correct_count + 1
                    # found_ngrams = set()
                    # for start_index, end_index, ngram, uri, freq, triple_count in subjects:
                    #     found_ngrams.add(ngram)
                    #
                    # print(text)
                    # print("Found ngrams: " + str(found_ngrams) + "\n")
                    #
                    # if correct_count == 30:
                    #     break
                # else:
                #     incorrect_count += 1
                #
                #     a = []
                #     if target_subject in inverted_mention_dict.keys():
                #         print(text)
                #         print(target_subject)
                #
                #         a = inverted_mention_dict[target_subject]
                #         print("Available mentions: " + str(a))
                #
                #         found_ngrams = set()
                #         for start_index, end_index, ngram, uri, freq, triple_count in subjects:
                #             found_ngrams.add(ngram)
                #
                #         print("Found ngrams: " + str(found_ngrams) + "\n")

            score = correct_count / float(total_count)
            avg_number_of_candidates = number_of_candidates / float(total_count)

            print("Dataset: " + d + " Upper bound: " + str(score))
            print("Average #candidates: " + str(avg_number_of_candidates))

            for k in recall_ranges:
                recall_at_k_score = recall_at_k[k] / float(total_count)
                print("\tRecall@" + str(k) + " : " + str(recall_at_k_score))
