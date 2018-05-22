import numpy as np
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from train.prepro import readfile,createBatches,createMatrices, addCharInformatioin,padding
from keras.initializers import RandomUniform
import os


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    for i,data in enumerate(dataset):
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, char], verbose=False)[0]
        pred = pred.argmax(axis=-1) #Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
    return predLabels, correctLabels

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



test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))
test_batch,test_batch_len = createBatches(test_set)


maxF1 = float(0)
maxEpoch = 0

for epoch in range(0, 100):
    path = "models/ner_model_epoch_"+str(epoch)+".hdf5"
    if os.path.isfile(path):
        model.load_weights(path)
        predLabels, correctLabels = tag_dataset(test_batch)
        f1_test = compute_f1(predLabels, correctLabels, idx2Label)

        if f1_test > maxF1:
            maxF1 = f1_test
            maxEpoch = epoch
        print("Epoch: " + str(epoch) + " F1: " + str(f1_test))





#   Performance on test dataset
print("Test-Data: Epoch: "+str(maxEpoch)+" F1: "+str(maxF1))