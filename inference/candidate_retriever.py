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


def get_stopwords():
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
    stopwords = list()
    for s in initial_stopwords:
        s = normalize_string(s)
        stopwords.append(s)
    return initial_stopwords


def is_invalid_span(new_start_index, new_end_index, candidates: list):
    for start_index, end_index, ngram, uri, freq in candidates:
        if start_index <= new_start_index and end_index >= new_end_index:

            ## if the bigger ngram start with "a", "the", "an", "of", "on", "at", "by"
            ### then allow the smaller ngram to be a candidate
            prefixes = ["a", "the", "an", "of", "on", "at", "by", "in"]
            for p in prefixes:
                if p == ngram.lower().split(" ")[0]:
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
        if i == len(ngram_tokens) - 1:
            ngram += token
        else:
            ngram += token + " "
    return ngram


def normalize_string(input):
    input = input.replace("\n", "").replace("'s", "")

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
    output = deaccent(input)
    output = unidecode.unidecode(output)
    return output


def extract_candidates_from_valid_ngram(sentence, dict, max_ngram_size, target_subject):
    tokens = sentence.split(" ")
    ngram_size = max_ngram_size
    candidates = list()
    while ngram_size > 0:
        extracted_ngrams = ngrams(tokens, ngram_size)
        for ngram_tokens in extracted_ngrams:

            start_index, end_index = span(ngram_tokens, tokens)

            ngram = combine_tokens(ngram_tokens)
            normalized_ngram = normalize_string(ngram)

            if normalized_ngram in dict.keys():

                matches = dict[normalized_ngram]

                has_correct_uri = False
                for uri in matches.keys():
                    if uri == target_subject:
                        has_correct_uri = True
                        break

                if has_correct_uri:
                    for uri in matches.keys():
                        freq = matches[uri]
                        candidates.append([start_index, end_index, uri, freq])

                    return candidates
        ##decrease the ngram size
        ngram_size = ngram_size - 1
    return None


def extract_candidates(ngram, dict, partial_match=False):
    normalized_ngram = normalize_string(ngram)

    candidates = list()

    if partial_match:
        for key in dict.keys:
            similarity_score = SequenceMatcher(None, key, normalized_ngram).ratio()

            if similarity_score >= 0.8:
                matches = dict[key]

                for uri in matches.keys():
                    freq = matches[uri]
                    candidates.append([uri, freq])
    else:
        if normalized_ngram in dict.keys():
            matches = dict[normalized_ngram]
            for uri in matches.keys():
                freq = matches[uri]
                candidates.append([uri, freq])

    return candidates

def extract_all_candidates(sentence, dict, max_ngram_size, stopwords, exclude_small_ngrams, exclude_stop_words):
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
                if is_invalid_span(start_index, end_index, candidates):
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

def load_freebase_index(file_path, stopwords):
    dict = {}

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

            if " / " in mention:
                mention = mention.split(" / ")[1]
            if ", " in mention:
                mention = mention.split(", ")[0]

            mention = normalize_string(mention)

            if mention in stopwords:
                continue


            if mention in dict.keys():
                uris = dict[mention]

                if uri in uris.keys():
                    uris[uri] +=1
                else:
                    uris[uri] = 1

                dict[mention] = uris
            else:
                uris = {}
                uris[uri] = 1
                dict[mention] = uris

    return dict

def load_subject_predicates(file_path):
    dict = {}

    with open(file_path, encoding="utf-8") as f:
        content = f.readlines()
        for entry in content:

            entries = entry.split("\t")
            if len(entries) < 2:
                continue

            uri = entries[0].replace("www.freebase.com/m/", "m.")
            predicate = entries[1].replace("www.freebase.com/", "").replace("/", ".")

            if uri in dict.keys():
                predicates = dict[uri]
                predicates.add(predicate)
                dict[uri] = predicates
            else:
                predicates = set()
                predicates.add(predicate)
                dict[uri] = predicates
    return dict


def generate_training_data():
    ## get stop words\
    stopwords = get_stopwords()

    print("Loading index")
    mention_dict = load_index("../data/surface_forms_new.txt")
    print("Loading all predicates for each subject")
    subject_predicate_dict = load_subject_predicates("../data/SimpleQuestions_v2/freebase-FB2M.txt")

    dataset_names = ["train"]
    max_ngram_size = 10

    for d in dataset_names:

        correct_count = 0
        dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            f1 = open('../data/ner_training_data.txt', 'w')
            f2 = open('../data/simple_qa_train_after_ner.txt', 'w')

            for i, line in enumerate(content):

                data = line.split("\t")
                target_subject = data[0].replace("www.freebase.com/m/", "m.")
                target_predicate = data[1].replace("www.freebase.com/", "").replace("/", ".")
                text = data[3].replace("\n", "")

                candidates = extract_candidates_from_valid_ngram(text, mention_dict, max_ngram_size, target_subject)

                # not found
                if candidates is None:
                    continue

                correct_count += 1

                tokens = text.split(" ")

                ## take the first element, since all elements have the same start and end index
                start_index, end_index, uri, freq = candidates[0]

                document = "-DOCSTART- O\n"
                for i, token in enumerate(tokens):
                    token = remove_accents(token)
                    token = strip_punctuation(token.lower())
                    if i == start_index or (i > start_index and i < end_index):
                        document += token + " I\n"
                    else:
                        document += token + " O\n"

                f1.write(document + '\n')  # python will convert \n to os.linesep

                candidates.sort(key=lambda tup: tup[3])  # sort by frequency
                subject_candidates = list()

                top_k = min(500, len(candidates))

                added_uris = set()
                for start_index, end_index, uri, freq in candidates[:top_k]:

                    if uri in added_uris:
                        continue

                    subject_candidate = {}
                    subject_candidate["startToken"] = int(start_index)
                    subject_candidate["endToken"] = int(end_index)
                    subject_candidate["uri"] = uri
                    subject_candidate["frequency"] = int(freq)

                    if uri not in subject_predicate_dict.keys():
                        continue

                    predicates = list()
                    for p in subject_predicate_dict[uri]:
                        predicates.append(p)

                    subject_candidate["predicates"] = predicates
                    subject_candidates.append(subject_candidate)

                entry = {}
                entry["text"] = text
                entry["subject"] = target_subject
                entry["predicate"] = target_predicate
                entry["candidates"] = subject_candidates

                f2.write(json.dumps(entry) + '\n')

            f1.close()
            f2.close()
        print("Found: " + str(correct_count / float(len(content))))

def extract_named_entities():
    ## get stop words\
    stopwords = get_stopwords()

    print("Loading index")
    mention_dict = load_freebase_index("../data/surface_forms.txt", stopwords)
    print("Loading all predicates for each subject")
    # subject_predicate_dict = load_subject_predicates("../data/SimpleQuestions_v2/freebase-FB2M.txt")

    dataset_names = ["test"]
    max_ngram_size = 10
    exclude_small_ngrams = True
    exclude_stopwords = True

    for d in dataset_names:

        correct_count = 0
        dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            f1 = open("../data/"+d+".txt", "w")

            for i, line in enumerate(content):

                data = line.split("\t")
                target_subject = data[0].replace("www.freebase.com/m/", "m.")
                target_predicate = data[1].replace("www.freebase.com/", "").replace("/", ".")
                text = data[3].replace("\n", "")

                candidates = extract_all_candidates(text, mention_dict, max_ngram_size, stopwords, exclude_small_ngrams, exclude_stopwords)

                # not found
                if candidates is None:
                    continue

                for start_index, end_index, ngram, uri, freq in candidates:
                    if uri == target_subject:
                        correct_count += 1




                # candidates.sort(key=lambda tup: tup[3])  # sort by frequency
                # subject_candidates = list()
                #
                # top_k = min(500, len(candidates))
                #
                # added_uris = set()
                # for start_index, end_index, uri, freq in candidates[:top_k]:
                #
                #     if uri in added_uris:
                #         continue
                #
                #     subject_candidate = {}
                #     subject_candidate["startToken"] = int(start_index)
                #     subject_candidate["endToken"] = int(end_index)
                #     subject_candidate["uri"] = uri
                #     subject_candidate["frequency"] = int(freq)
                #
                #     if uri not in subject_predicate_dict.keys():
                #         continue
                #
                #     predicates = list()
                #     for p in subject_predicate_dict[uri]:
                #         predicates.append(p)
                #
                #     subject_candidate["predicates"] = predicates
                #     subject_candidates.append(subject_candidate)
                #
                # entry = {}
                # entry["text"] = text
                # entry["subject"] = target_subject
                # entry["predicate"] = target_predicate
                # entry["candidates"] = subject_candidates
                #
                # f1.write(json.dumps(entry) + '\n')

            f1.close()
        print("Found: " + str(correct_count / float(len(content))))

extract_named_entities()