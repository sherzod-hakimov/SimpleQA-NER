import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding
from keras.utils import Progbar
from keras.initializers import RandomUniform
import os.path
from extract_all_words import extract_words
from candidate_retriever import generate_training_data

epochs = 100
training_data_path  = "../data/ner_training_data.txt"
all_words_path  = "../data/words.txt"
word_embedding_path ="../data/glove.6B.100d.txt"

if not os.path.isfile(all_words_path):
    extract_words()

if not os.path.isfile(training_data_path):
    generate_training_data()

trainSentences = readfile(training_data_path)
trainSentences = addCharInformatioin(trainSentences)


##LOAD all words from train, test and dev
words = {}
with open(all_words_path, encoding="utf-8") as f:
    content = f.readlines()
    for w in enumerate(content):
        words[w] = True

# :: Create a mapping for the labels ::
label2Idx = {}
label2Idx["I"] =1
label2Idx["O"] =0


# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open(word_embedding_path, encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
    char2Idx[c] = len(char2Idx)


# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

train_set = padding(createMatrices(trainSentences,word2Idx, label2Idx, case2Idx,char2Idx))

train_batch,train_batch_len = createBatches(train_set)


words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                  trainable=False)(words_input)
casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)
character_input = Input(shape=(None, 52,), name='char_input')
embed_char_out = TimeDistributed(
    Embedding(len(char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(
    character_input)
dropout = Dropout(0.5)(embed_char_out)
conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words,casing, char])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()


for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels, token_input, case_input, char_input = batch
        model.train_on_batch([token_input, case_input, char_input], labels)
        a.update(i)
        print(' ')
# save the model
model.save_weights("../data/ner_model.hdf5", overwrite=True)


print("Finished train the NER model!")