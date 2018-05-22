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


def normalize_string(input, stemmer):
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
    output = deaccent(input)
    output = unidecode.unidecode(output)
    return output


def extract_valid_ngram(sentence, dict, max_ngram_size, stemmer, target_subject):
    tokens = sentence.split(" ")
    ngram_size = max_ngram_size
    candidates = list()
    while ngram_size > 0:
        extracted_ngrams = ngrams(tokens, ngram_size)
        for ngram_tokens in extracted_ngrams:

            start_index, end_index = span(ngram_tokens, tokens)

            ngram = combine_tokens(ngram_tokens)
            normalized_ngram = normalize_string(ngram, stemmer)

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

def extract_subject(sentence, dict, max_ngram_size, stemmer, target_subject):
    tokens = sentence.split(" ")
    ngram_size = max_ngram_size
    while ngram_size > 0:
        extracted_ngrams = ngrams(tokens, ngram_size)
        for ngram_tokens in extracted_ngrams:

            start_index, end_index = span(ngram_tokens, tokens)

            ngram = combine_tokens(ngram_tokens)
            normalized_ngram = normalize_string(ngram, stemmer)

            if normalized_ngram in dict.keys():

                matches = dict[normalized_ngram]

                for uri in matches.keys():
                    if uri == target_subject:
                        return start_index, end_index
        ##decrease the ngram size
        ngram_size = ngram_size - 1
    return 0,0

def load_index(file_path, stopwords, stemmer):
    dict = {}
    inverted_dict = {}

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
                added_mentions.add(mention)
                inverted_dict[uri] = added_mentions
            else:
                added_mentions = set()
                added_mentions.add(mention)
                inverted_dict[uri] = added_mentions

    return dict, inverted_dict



def generate_training_data():
    stemmer = SnowballStemmer("english")

    ## get stop words and normalize
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
        s = normalize_string(s, stemmer)
        stopwords.append(s)

    print("Loading index")
    mention_dict, inverted_mention_dict = load_index("../data/surface_forms.txt", stopwords, stemmer)

    dataset_names = ["train"]
    max_ngram_size = 10

    for d in dataset_names:

        correct_count = 0
        dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            f = open('../data/ner_training_data.txt', 'w')

            for i, line in enumerate(content):

                data = line.split("\t")
                target_subject = data[0].replace("www.freebase.com/m/", "m.")
                target_predicate = data[1].replace("www.freebase.com/", "")
                text = data[3].replace("\n", "")

                candidates = extract_valid_ngram(text, mention_dict, max_ngram_size, stemmer, target_subject)

                #not found
                if candidates is None:
                    continue

                correct_count +=1

                tokens = text.split(" ")

                ## take the first element, since all elements have the same start and end index
                start_index, end_index, uri, freq = candidates[0]

                document = "-DOCSTART- O\n"
                for i, token in enumerate(tokens):
                    token = remove_accents(token)
                    token = strip_punctuation(token.lower())
                    if i==start_index or (i> start_index and i<end_index):
                        document +=token+" I\n"
                    else:
                        document += token + " O\n"

                f.write(document + '\n')  # python will convert \n to os.linesep
            f.close()  # you can omit in most cases as the destructor will call it
        print("Found: "+str(correct_count/float(len(content))))

def extract_candidates():
    stemmer = SnowballStemmer("english")

    ## get stop words and normalize
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
        s = normalize_string(s, stemmer)
        stopwords.append(s)

    print("Loading index")
    mention_dict, inverted_mention_dict = load_index("../data/surface_forms.txt", stopwords, stemmer)

    dataset_names = ["test", "valid"]
    max_ngram_size = 10

    for d in dataset_names:

        correct_count = 0
        dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

        with open(dataset_path, encoding="utf-8") as f:
            content = f.readlines()

            output_file_path = "../data/"+d+"_before_ner.txt"

            f = open(output_file_path, 'w')

            for i, line in enumerate(content):

                data = line.split("\t")
                target_subject = data[0].replace("www.freebase.com/m/", "m.")
                target_predicate = data[1].replace("www.freebase.com/", "")
                text = data[3].replace("\n", "")

                # text = strip_punctuation(text.lower())
                start_index, end_index = extract_valid_ngram(text, mention_dict, max_ngram_size, stemmer, target_subject)

                #not found
                if start_index ==0 and end_index == 0:
                    continue

                correct_count +=1

                tokens = text.split(" ")

                document = "-DOCSTART- O\n"
                for i, token in enumerate(tokens):
                    token = remove_accents(token)
                    token = strip_punctuation(token.lower())
                    if i==start_index or (i> start_index and i<end_index):
                        document +=token+" I\n"
                    else:
                        document += token + " O\n"

                f.write(document + '\n')  # python will convert \n to os.linesep
            f.close()  # you can omit in most cases as the destructor will call it
        print("Found: "+str(correct_count/float(len(content))))