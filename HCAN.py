
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import time
import datetime
import string
from tensorflow.contrib.layers import layer_norm
from nltk import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split

datapath = 'F:/UChicago/degree paper/Archive/' # Datapath where the data is saved
savedir  = 'F:/UChicago/degree paper/Archive/save/'      # Savepath to save the result from the model


# Load sample dataset
#npz = np.load(datapath + 'yelp_review_small.npz', allow_pickle=True)
#npz = np.load(datapath + 'yelp_review.npz', allow_pickle=True)
npz = np.load(datapath + 'amazon_review_100k.npz', allow_pickle=True)
data = npz['arr_0']
data[:3,]


X = data[:,0] # Text data
Y = data[:,1] # Label
del data


# Transfer Label to Onehot encoding
le = LabelEncoder()
y = le.fit_transform(Y)
lb = LabelBinarizer()
lb.fit(y)
LabelEncoding = lb.transform(y)


# (Train, Valid, Test)=(0.8, 0.1, 0.1)
X_train,X_test,y_train,y_test = train_test_split(X,LabelEncoding,
                                                 test_size=0.2,random_state=14, stratify = LabelEncoding)
X_valid,X_test,y_valid,y_test = train_test_split(X_test,y_test,
                                                test_size=0.5,random_state=14,stratify=y_test)


# Precomputed word2idx dictionary
#with open('%sword2idx_yelp_small.json' % datapath) as f:
#with open('%sword2idx_yelp.json' % datapath) as f:
with open('%sword2idx_amazon_100k.json' % datapath) as f:
    w2i = json.load(f)

# Precomputed wordembedding matrix with dimension 50
#npz = np.load('%sWordEmbedding_yelp_small.npz' % datapath)
#npz = np.load('%sWordEmbedding_yelp.npz' % datapath)
npz = np.load('%sWordEmbedding_amazon_100k.npz' % datapath)
WordEmbeddings = npz['arr_0']


# Remove puntuation
def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation + "\n"))


# Convert sentence to word
def ConvertSentence2Word(s):
    return (word_tokenize(remove_punctuation(s).lower()))


# Convert sentence to word idx
def ConvertSent2Idx(s):
    s_temp = [w for w in ConvertSentence2Word(s) if w in w2i]
    temp = []
    for w in s_temp:
        temp.append(w2i[w])
    return (temp)


# Divide Document with variable length of sentences
def ConvertDoc2List(doc):
    temp_doc = sent_tokenize(doc)
    temp = []
    for i in range(len(temp_doc)):
        if (len(ConvertSent2Idx(temp_doc[i])) >= 1):  # Prevent empty sentence
            temp.append(ConvertSent2Idx(temp_doc[i]))
    return (temp)


# Convert List type of document to array
def ConvertList2Array(docs):
    ms = len(docs)
    mw = len(max(docs, key=len))
    result = np.zeros((ms, mw))
    for i, line in enumerate(docs):
        for j, word in enumerate(line):
            result[i, j] = word
    return result


# Reset tensor graph
def reset_graph(seed=14):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Maximum number of word in sentence
max_words = 0
max_sents = 0
num_docs = len(Y)
start = time.time()
for i in range(num_docs):
    sys.stdout.write("processing record %i of %i       \r" % (i+1,num_docs))
    sys.stdout.flush()
    sents = sent_tokenize(X[i])
    if(len(sents)>max_sents):
        max_sents=len(sents)
    for sent in sents:
        temp = len([w for w in ConvertSentence2Word(sent) if w in w2i])
        if temp > max_words:
            max_words = temp
print("\nTime: %.2f" % (time.time()-start))


num_classes = LabelEncoding.shape[1]
attention_size = WordEmbeddings.shape[1]
attention_head = 5
activation = tf.nn.elu
dropout_rate = 0.9
n_epochs = 30
he_init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float64)
xavier_init = tf.contrib.layers.xavier_initializer(dtype=tf.float64)


reset_graph()
dropout = tf.placeholder(tf.float64)
document = tf.placeholder(tf.int64, shape=[None, None])
ws = tf.reduce_sum(tf.sign(document),1)     # Number of words per sentence
ms = tf.reduce_sum(tf.sign(ws))             # Number of sentences in documents
mw = tf.reduce_max(ws)                      # Maximum number of words in sentence

EmbeddingMatrix = tf.get_variable('WordEmbeddings', initializer=WordEmbeddings, dtype=tf.float64)
word_embeds = tf.nn.embedding_lookup(EmbeddingMatrix, document)    # Lookup table


positions = tf.expand_dims(tf.range(mw),0)  # Vector to indicate the location of words
PositionalEmbeddings = tf.get_variable('WordPosition', shape = (max_words,attention_size),
                                       initializer= xavier_init, dtype = tf.float64)
pos_embeds = tf.gather(PositionalEmbeddings, positions)
word_embeds = tf.nn.dropout(word_embeds+pos_embeds, dropout)


#word self attention 1
Q1 = tf.layers.conv1d(word_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
K1 = tf.layers.conv1d(word_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
V1 = tf.layers.conv1d(word_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)

Q1 = tf.concat(tf.split(Q1,attention_head,axis=2),axis=0)
K1 = tf.concat(tf.split(K1,attention_head,axis=2),axis=0)
V1 = tf.concat(tf.split(V1,attention_head,axis=2),axis=0)

outputs1 = tf.matmul(Q1,tf.transpose(K1,[0, 2, 1]))
outputs1 = outputs1/(K1.get_shape().as_list()[-1]**0.5)
outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1),dropout)
outputs1 = tf.matmul(outputs1,V1)
outputs1 = tf.concat(tf.split(outputs1,attention_head,axis=0),axis=2)


#word self attention 2
Q2 = tf.layers.conv1d(word_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
K2 = tf.layers.conv1d(word_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
V2 = tf.layers.conv1d(word_embeds,attention_size,3,padding='same',
                      activation=tf.nn.tanh,kernel_initializer=he_init)

Q2 = tf.concat(tf.split(Q2,attention_head,axis=2),axis=0)
K2 = tf.concat(tf.split(K2,attention_head,axis=2),axis=0)
V2 = tf.concat(tf.split(V2,attention_head,axis=2),axis=0)

outputs2 = tf.matmul(Q2,tf.transpose(K2,[0, 2, 1]))
outputs2 = outputs2/(K2.get_shape().as_list()[-1]**0.5)
outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2),dropout)
outputs2 = tf.matmul(outputs2,V2)
outputs2 = tf.concat(tf.split(outputs2,attention_head,axis=0),axis=2)


outputs = tf.math.multiply(outputs1,outputs2) # Elementwise multiply
outputs = layer_norm(outputs)                 # Layer norm


#word target attention
T = tf.get_variable('WordTarget',(1,1,attention_size), tf.float64, he_init)
T = tf.tile(T,[ms,1,1])
K = tf.layers.conv1d(outputs,attention_size,3,padding='same',
                     activation=activation,kernel_initializer=he_init)
V = tf.layers.conv1d(outputs,attention_size,3,padding='same',
                     activation=activation,kernel_initializer=he_init)

T = tf.concat(tf.split(T,attention_head,axis=2),axis=0)
K = tf.concat(tf.split(K,attention_head,axis=2),axis=0)
V = tf.concat(tf.split(V,attention_head,axis=2),axis=0)

sent_embeds = tf.matmul(T,tf.transpose(K,[0, 2, 1]))
sent_embeds = sent_embeds/(K.get_shape().as_list()[-1]**0.5)
sent_embeds = tf.nn.dropout(tf.nn.softmax(sent_embeds),dropout)
sent_embeds = tf.matmul(sent_embeds, V)
sent_embeds = tf.concat(tf.split(sent_embeds,attention_head,axis=0),axis=2)
sent_embeds = tf.transpose(sent_embeds, [1,0,2])


positions = tf.expand_dims(tf.range(ms),0)
PositionalEmbeddings = tf.get_variable('SentPosition', shape = (max_sents,attention_size),
                                       initializer= xavier_init, dtype = tf.float64)
pos_embeds = tf.gather(PositionalEmbeddings, positions)
sent_embeds = tf.nn.dropout(sent_embeds+pos_embeds, dropout)


Q1 = tf.layers.conv1d(sent_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
K1 = tf.layers.conv1d(sent_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
V1 = tf.layers.conv1d(sent_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)

Q1 = tf.concat(tf.split(Q1,attention_head,axis=2),axis=0)
K1 = tf.concat(tf.split(K1,attention_head,axis=2),axis=0)
V1 = tf.concat(tf.split(V1,attention_head,axis=2),axis=0)

outputs1 = tf.matmul(Q1,tf.transpose(K1,[0, 2, 1]))
outputs1 = outputs1/(K1.get_shape().as_list()[-1]**0.5)
outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1),dropout)
outputs1 = tf.matmul(outputs1,V1)
outputs1 = tf.concat(tf.split(outputs1,attention_head,axis=0),axis=2)


Q2 = tf.layers.conv1d(sent_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
K2 = tf.layers.conv1d(sent_embeds,attention_size,3,padding='same',
                      activation=activation,kernel_initializer=he_init)
V2 = tf.layers.conv1d(sent_embeds,attention_size,3,padding='same',
                      activation=tf.nn.tanh,kernel_initializer=he_init)

Q2 = tf.concat(tf.split(Q2,attention_head,axis=2),axis=0)
K2 = tf.concat(tf.split(K2,attention_head,axis=2),axis=0)
V2 = tf.concat(tf.split(V2,attention_head,axis=2),axis=0)

outputs2 = tf.matmul(Q2,tf.transpose(K2,[0, 2, 1]))
outputs2 = outputs2/(K2.get_shape().as_list()[-1]**0.5)
outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2),dropout)
outputs2 = tf.matmul(outputs2,V2)
outputs2 = tf.concat(tf.split(outputs2,attention_head,axis=0),axis=2)


outputs = tf.multiply(outputs1,outputs2)
outputs = layer_norm(outputs)


T = tf.get_variable('SentTarget',(1,1,attention_size), tf.float64,he_init)
K = tf.layers.conv1d(outputs,attention_size,3,padding='same',
                     activation=activation,kernel_initializer=he_init)
V = tf.layers.conv1d(outputs,attention_size,3,padding='same',
                     activation=activation,kernel_initializer=he_init)

T = tf.concat(tf.split(T,attention_head,axis=2),axis=0)
K = tf.concat(tf.split(K,attention_head,axis=2),axis=0)
V = tf.concat(tf.split(V,attention_head,axis=2),axis=0)

doc_embed = tf.matmul(T,tf.transpose(K,[0, 2, 1]))
doc_embed = doc_embed/(K.get_shape().as_list()[-1]**0.5)
doc_embed = tf.nn.dropout(tf.nn.softmax(doc_embed),dropout)
doc_embed = tf.matmul(doc_embed, V)
doc_embed = tf.concat(tf.split(doc_embed,attention_head,axis=0),axis=2)
doc_embed = tf.squeeze(doc_embed,[0])


output = tf.layers.dense(doc_embed,num_classes,kernel_initializer=xavier_init)
prediction = tf.nn.softmax(output)
prediction = tf.nn.dropout(prediction, dropout)


def val_score(valdata, vallabels):
    correct = 0
    for i in range(len(valdata)):
        X_input = ConvertDoc2List(valdata[i])
        if len(X_input) < 1:
            continue
        X_input = ConvertList2Array(X_input)
        feed_dict = {document: X_input, dropout: 1.0}
        val_pred = sess.run(prediction, feed_dict=feed_dict)
        if np.argmax(val_pred) == np.argmax(vallabels[i]):
            correct +=1
        val_acc = correct/len(valdata)
    return val_acc


label = tf.placeholder(tf.float64, shape=[num_classes])
labels = tf.expand_dims(label,0)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=labels))
#optimizer = tf.train.AdamOptimizer(2e-5,0.9,0.99)
optimizer   = tf.train.GradientDescentOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
start = time.time()
end = time.time()
saver = tf.train.Saver()
bestscore = 0
train_acc = []
valid_acc = []
X_input_data = []
y_input_data = []
# First we will process the data, transfer them to array
for i in range(len(X_train)):
    X_input = ConvertDoc2List(X_train[i])
    if(len(X_input)<1):
        continue
    X_input = ConvertList2Array(X_input)
    y_input = y_train[i]
    X_input_data.append(X_input)
    y_input_data.append(y_input)

with tf.Session() as sess:
    init.run()
    correct = 0
    for epoch in range(n_epochs):
        for i in range(len(X_input_data)):
            # X_input = ConvertDoc2List(X_train[i])
            # if (len(X_input) < 1):  # Prevent Empty Document
            #    continue
            # X_input = ConvertList2Array(X_input)
            # y_input = y_train[i]
            X_input = X_input_data[i]
            y_input = y_input_data[i]
            feed_dict = {document: X_input, label: y_input, dropout: dropout_rate}
            pred, cost, _ = sess.run([prediction, loss, training_op], feed_dict=feed_dict)

            if np.argmax(pred) == np.argmax(y_input):
                correct += 1
            #sys.stdout.write("epoch %i, sample %i of %i, loss: %f\r" % (epoch + 1, i + 1, len(X_train), cost))
            #sys.stdout.flush()

        trainscore = correct / len(X_train)
        valscore = val_score(X_valid, y_valid)

        train_acc.append(trainscore)
        valid_acc.append(valscore)

        if valscore >= bestscore:
            bestscore = valscore
            #save_path = saver.save(sess, savedir + "savedmodels/hcan_small_50_ep8.ckpt")
            #save_path = saver.save(sess, savedir + "savedmodels/hcan_50_ep8.ckpt")
            save_path = saver.save(sess, savedir + "savedmodels/hcan_50_amazon.ckpt")
        temptime = datetime.timedelta(seconds=round(time.time() - start))
        print("epoch %i, training accuracy: %.2f, validation accuracy: %.2f," % (
        epoch + 1, trainscore * 100, valscore * 100), "time: ", temptime)
        correct = 0
        start = time.time()
totaltime = datetime.timedelta(seconds=round(time.time() - end))
print("\nTime:", totaltime)


# Save Accuracy
accuracy = np.column_stack((np.array(train_acc), np.array(valid_acc)))
#np.savez(datapath + 'accuracy_hcan_small_50_ep8.npz', accuracy )
#np.savez(datapath + 'accuracy_hcan_50_ep8.npz', accuracy )
np.savez(datapath + 'accuracy_hcan_50_amazon.npz', accuracy )


# Plot the accuracy
epochs = range(1,n_epochs+1)
plt.plot(epochs, train_acc, label="train")
plt.plot(epochs, valid_acc, label="valid")
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


def test_score(data, label=[]):
    labels = []
    init_op = tf.global_variables_initializer()
    correct = 0

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, save_path)
        for i in range(len(data)):
            X_input = ConvertDoc2List(data[i])
            if len(X_input) < 1:
                continue
            X_input = ConvertList2Array(X_input)
            feed_dict = {document: X_input, dropout: 1.0}
            pred = sess.run(prediction, feed_dict=feed_dict)

            if np.argmax(pred) == np.argmax(label[i]):
                correct += 1
    return correct / len(data)


start = time.time()
testscore = test_score(X_test, y_test)
totaltime = datetime.timedelta(seconds=round(time.time()-start))
print("\ntest accuracy: %.2f" % (testscore*100),"%")
print("Time:", totaltime)
