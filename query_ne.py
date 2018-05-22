import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from train.prepro import readfile, createMatrices, addCharInformatioin,padding
from keras.utils import Progbar
from keras.initializers import RandomUniform
import json
import requests
from urllib.parse import quote

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, char], verbose=False)[0]
        pred = pred.argmax(axis=-1) #Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

def query(ngram, sim_threshold):
    content = []

    ngram = quote(ngram)

    try:
        response = requests.get("http://purpur-v10:8080/findEntities?query="+ngram+"&k=100")
        content = json.loads(response.content.decode('utf-8'))
    except json.decoder.JSONDecodeError as e:
        print("NGram: "+ngram)
        print(e)

    if len(content) == 0:
        try:
            response = requests.get("http://purpur-v10:8080/findEntitiesWithPartialMatch?query="+ngram+"&k=100&minSim="+str(sim_threshold))
            content = json.loads(response.content.decode('utf-8'))
        except json.decoder.JSONDecodeError as e:
            print("NGram: " + ngram)
            print(e)
    return content

trainSentences = readfile("data/train.txt")
devSentences = readfile("data/valid.txt")
testSentences = readfile("data/test.txt")

trainSentences = addCharInformatioin(trainSentences)
devSentences = addCharInformatioin(devSentences)
testSentences = addCharInformatioin(testSentences)

labelSet = set()
words = {}

for dataset in [trainSentences, devSentences, testSentences]:
    for sentence in dataset:
        for token, char, label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
label2Idx["I"] =1
label2Idx["O"] =0
# for label in labelSet:
#     label2Idx[label] = len(label2Idx)

# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
            'contains_digit': 6, 'PADDING_TOKEN': 7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]

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


##MODEL
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


path = "models/ner_model_epoch_70.hdf5"
model.load_weights(path)


with open('data/test_all_ngrams.txt') as f:
    content = f.readlines()
    f = open('data/test_filtered.txt', 'w')

    upper_bound_count = 0
    empty_prediction_count = 0
    empty_candidate_count = 0
    total_count = 0

    for i, line in enumerate(content):

        total_count+=1

        # if i == 1000:
        #     break

        json_data = json.loads(line)
        text = json_data["text"]
        words = text.split(' ')

        tokens = []
        for w in words:
            tokens.append([w, 'O'])

        testSentences = []
        testSentences.append(tokens)
        testSentences = addCharInformatioin(testSentences)
        test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx, char2Idx))

        tokens, casing, char, labels = test_set[0]
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
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

        filteredCandidates = []

        if predicted_span == "":
            empty_prediction_count += 1
        else:
            candidates = query(predicted_span, 0.7)
            if len(candidates) > 0:
                filteredCandidates = candidates
            is_found = False
            for c in candidates:
                if c["uri"] == json_data["subject"]:
                    upper_bound_count += 1
                    is_found = True
                    break
            if not is_found:
                print(text+"-> "+predicted_span +" : " + json_data['subject'])

        ##just increment if the expected is in the list
        for c in filteredCandidates:
            if json_data["subject"] == None:
                print(json_data)
            if c["uri"] == json_data["subject"]:
                upper_bound_count+=1
                break

        filteredLine = {}

        filteredLine['text'] = json_data['text']
        filteredLine['subject'] = json_data['subject']
        filteredLine['predicate'] = json_data['predicate']
        filteredLine['candidates'] = filteredCandidates

        # f.write(json.dumps(filteredLine) + '\n')  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it

    upper_boud = upper_bound_count/float(total_count)

    print("Upper bound: "+str(upper_boud))
    print("Empty prediction: "+str(empty_prediction_count))
    print("Empty candidate: " + str(empty_candidate_count))