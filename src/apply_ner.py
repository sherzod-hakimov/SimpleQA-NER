import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate

from src.candidate_retriever import load_index, extract_candidates, remove_accents, strip_punctuation
from src.extract_all_words import extract_words
from src.prepro import createMatrices, addCharInformatioin, padding
from keras.initializers import RandomUniform
import json
import os.path

epochs = 100
all_words_path = "../data/words.txt"
word_embedding_path = "../data/glove.6B.100d.txt"

if not os.path.isfile(all_words_path):
    extract_words()

##LOAD all words from train, test and dev
words = {}
with open(all_words_path, encoding="utf-8") as f:
    content = f.readlines()
    for w in enumerate(content):
        words[w] = True

# :: Create a mapping for the labels ::
label2Idx = {}
label2Idx["I"] = 1
label2Idx["O"] = 0

# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open(word_embedding_path, encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")

    if len(word2Idx) == 0:  # Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)

        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)

wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {"PADDING": 0, "UNKNOWN": 1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
    char2Idx[c] = len(char2Idx)

idx2Label = {v: k for k, v in label2Idx.items()}

words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                  trainable=False)(words_input)
character_input = Input(shape=(None, 52,), name='char_input')
embed_char_out = TimeDistributed(
    Embedding(len(char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(
    character_input)
dropout = Dropout(0.5)(embed_char_out)
conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, char])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input, character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()

path = "../data/ner_model.hdf5"
model.load_weights(path)

mention_dict = load_index("../data/surface_forms_new.txt")
dataset_names = ["test"]

for d in dataset_names:

    dataset_path = "../data/SimpleQuestions_v2/annotated_fb_data_" + d + ".txt"

    recall_ranges = [1, 5, 10, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 10000]
    recall_at_k = {}

    ##initialize with 0 count
    for r in recall_ranges:
        recall_at_k[r] = 0

    with open(dataset_path, encoding="utf-8") as f:
        content = f.readlines()

        correct_count = 0

        f = open("../data/" + d + "_after_ner.txt", "w")

        for i, line in enumerate(content):

            is_found = False
            data = line.split("\t")
            target_subject = data[0].replace("www.freebase.com/m/", "m.")
            target_predicate = data[1].replace("www.freebase.com/", "").replace("/", ".")
            text = data[3].replace("\n", "")

            ### prepare for prediction
            words = text.split(' ')

            tokens = []
            for w in words:
                w = remove_accents(w)
                w = strip_punctuation(w.lower())
                tokens.append([w, 'O'])

            testSentences = []
            testSentences.append(tokens)
            testSentences = addCharInformatioin(testSentences)
            test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, char2Idx))

            tokens, char, labels = test_set[0]
            tokens = np.asarray([tokens])
            char = np.asarray([char])

            ## PREDICT
            prediction = model.predict([tokens, char], verbose=False)[0]
            prediction = prediction.argmax(axis=-1)  # Predict the classes

            predicted_span = ""
            start_index = -1
            end_index = -1
            for token_index, pred_label in enumerate(prediction):
                if pred_label == 1:  ## I
                    if start_index == -1:
                        start_index = token_index
                    end_index = token_index + 1
                    predicted_span += words[token_index] + " "
            predicted_span = predicted_span.strip()



            candidates = extract_candidates(predicted_span, mention_dict, partial_match=False)
            ## sort
            candidates.sort(key=lambda tup: tup[1])  # sorts in place

            ### Evaluate Recall@K
            ### crop the list and compare recall@k
            for range in recall_ranges:
                ##no need to crop again if it's found on prev k number
                if is_found:
                    recall_at_k[range] = recall_at_k[range] + 1
                    continue

                top_k = min(range, len(candidates))
                filtered_subjects = candidates[:top_k]

                for u1, f1 in filtered_subjects:
                    if u1 == target_subject:
                        is_found = True
                        break

                if is_found:
                    recall_at_k[range] = recall_at_k[range] + 1



            subject_candidates = list()

            is_found = False
            for uri, freq in candidates:
                subject_candidate = {}
                subject_candidate["startToken"] = start_index
                subject_candidate["endToken"] = end_index
                subject_candidate["predicates"] = list()
                subject_candidate["uri"] = uri
                subject_candidate["frequency"] = freq

                subject_candidates.append(subject_candidate)

                if target_subject == uri:
                    correct_count +=1
                    is_found = True

            if not is_found:
                print("Text: " + text + " NE: " + predicted_span + "\n")

            entry = {}
            entry["text"] = text
            entry["predicate"] = target_predicate
            entry["subject"] = target_subject
            entry["candidates"] = subject_candidates

            f.write(json.dumps(entry) + '\n')  # python will convert \n to os.linesep
        f.close()

        print("Correct predicted span: "+ str(correct_count/float(len(content))))
        for k in recall_ranges:
            recall_at_k_score = recall_at_k[k] / float(len(content))
            print("\tRecall@" + str(k) + " : " + str(recall_at_k_score))

# with open('data/test_all_ngrams.txt') as f:
#     content = f.readlines()
#     f = open('data/test_filtered.txt', 'w')
#
#     upper_bound_count = 0
#     empty_prediction_count = 0
#     empty_candidate_count = 0
#
#     for i, line in enumerate(content):
#
#         json_data = json.loads(line)
#         text = json_data["text"]
#         candidates = json_data['candidates']
#         filteredCandidates = []
#
#         if len(candidates) == 0:
#             empty_candidate_count +=1
#
#         if len(candidates) == 1:
#             filteredCandidates.append(candidates[0])
#         else:
#             words = text.split(' ')
#
#             tokens = []
#             for w in words:
#                 tokens.append([w, 'O'])
#
#             testSentences = []
#             testSentences.append(tokens)
#             testSentences = addCharInformatioin(testSentences)
#             test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, char2Idx))
#
#
#             tokens, char, labels = test_set[0]
#             tokens = np.asarray([tokens])
#             char = np.asarray([char])
#             prediction = model.predict([tokens, char], verbose=False)[0]
#             prediction = prediction.argmax(axis=-1)  # Predict the classes
#
#             predicted_span = ""
#             start_index = -1
#             end_index = -1
#             for token_index, pred_label in enumerate(prediction):
#                 if pred_label == 1:  ## I
#                     if start_index == -1:
#                         start_index = token_index
#                     end_index = token_index + 1
#                     predicted_span += words[token_index] + " "
#             predicted_span = predicted_span.strip()
#
#             if predicted_span == "":
#                 empty_prediction_count+=1
#
#
#             for c in candidates:
#                 if (c['startToken'] == start_index and c['endToken'] == end_index):
#                     if len(c['predicates']) > 0:
#                         filteredCandidates.append(c)
#
#             ## find the most similar if it doesn't match any candidate span
#             max_similarity_score = 0
#             max_candidate = None
#             if len(filteredCandidates) == 0:
#                 for c in candidates:
#                     if len(c['predicates']) > 0:
#
#                         nGram = json.dumps(c['ngram'])
#                         firstString = nGram.strip().replace("\"", "")
#                         secondString = predicted_span.strip().replace("\"", "")
#                         similarity_score = SequenceMatcher(None, firstString, secondString).ratio()
#
#                         if (similarity_score > max_similarity_score):
#                             max_similarity_score = similarity_score
#                             max_candidate = c
#                 if max_candidate != None:
#                     filteredCandidates.append(max_candidate)
#
#             ## find the max ngram if it doesn't match any candidate span
#             max_ngram = 0
#             max_candidate = None
#             if len(filteredCandidates) == 0:
#                 for c in candidates:
#                     if max_ngram < c["ngramSize"]:
#                         max_candidate = c
#                         max_ngram = c["ngramSize"]
#                 if max_candidate!=None:
#                     filteredCandidates.append(max_candidate)
#
#         ##just increment if the expected is in the list
#         for c in filteredCandidates:
#             if json_data["subject"] == None:
#                 print(json_data)
#             if c["uri"] == json_data["subject"]:
#                 upper_bound_count+=1
#                 break
#
#         filteredLine = {}
#
#         filteredLine['text'] = json_data['text']
#         filteredLine['subject'] = json_data['subject']
#         filteredLine['predicate'] = json_data['predicate']
#         filteredLine['candidates'] = filteredCandidates
#
#         f.write(json.dumps(filteredLine) + '\n')  # python will convert \n to os.linesep
#     f.close()  # you can omit in most cases as the destructor will call it
#
#     upper_boud = upper_bound_count/float(len(content))
#
#     print("Upper bound: "+str(upper_boud))
#     print("Empty prediction: "+str(empty_prediction_count))
#     print("Empty candidate: " + str(empty_candidate_count))
