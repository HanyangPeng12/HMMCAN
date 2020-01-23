
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import time
import datetime
import string
from nltk import sent_tokenize, word_tokenize
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.contrib.rnn import GRUCell
import keras

datapath = 'F:/UChicago/degree paper/Archive/' # Datapath where the data is saved
savedir  = 'F:/UChicago/degree paper/Archive/save/'      # Savepath to save the result from the model


# Load sample dataset
#npz = np.load(datapath + 'yelp_review_small.npz', allow_pickle=True)
npz = np.load(datapath + 'yelp_review.npz', allow_pickle=True)
#npz = np.load(datapath + 'amazon_review_100k.npz', allow_pickle=True)
data = npz['arr_0']
data[:3,]


X = data[:,0]  # Text
Y = data[:,1]  # Label
del data


# Transfer Label to Onehot encoding
le = LabelEncoder()
y = le.fit_transform(Y)
lb = LabelBinarizer()
lb.fit(y)
LabelEncoding = lb.transform(y)


# (Train, Valid, Test)=(0.8, 0.1, 0.1)
X_train,X_test,y_train,y_test = train_test_split(X,LabelEncoding, test_size=0.2,random_state=14, stratify = LabelEncoding)
X_valid,X_test,y_valid,y_test = train_test_split(X_test,y_test, test_size=0.5,random_state=14,stratify=y_test)


# Precomputed word2idx dictionary
#with open('%sword2idx_yelp_small.json' % datapath) as f:
with open('%sword2idx_yelp.json' % datapath) as f:
#with open('%sword2idx_amazon_100k.json' % datapath) as f:
    w2i = json.load(f)

# Precomputed wordembedding matrix with dimension 50
#npz = np.load('%sWordEmbedding_yelp_small.npz' % datapath)
npz = np.load('%sWordEmbedding_yelp.npz' % datapath)
#npz = np.load('%sWordEmbedding_amazon_100k.npz' % datapath)
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


num_classes = LabelEncoding.shape[1]
attention_dim = WordEmbeddings.shape[1]
rnn_dim = WordEmbeddings.shape[1]   # Dimension of hidden layer
dropout_rate = 0.9
n_epochs = 30
xavier_init = tf.contrib.layers.xavier_initializer(dtype=tf.float64)


reset_graph()

dropout = tf.placeholder(tf.float64)
document = tf.placeholder(tf.int32, shape=[None,None])
words_sent = tf.reduce_sum(tf.sign(document),1)       # Number of words per sentence
num_sent = tf.reduce_sum(tf.sign(words_sent))         # Number of sentences in documents
num_word = tf.reduce_max(words_sent)                  # Maximum number of words in sentence

EmbeddingMatrix = tf.get_variable('WordEmbeddings', initializer=WordEmbeddings, dtype=tf.float64)
word_embeds = tf.nn.embedding_lookup(EmbeddingMatrix, document)

# word encoding
with tf.variable_scope('words'):
    [FowardEmbedding, BackwardEmbedding], _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.DropoutWrapper(GRUCell(rnn_dim), state_keep_prob=dropout),
        cell_bw=tf.contrib.rnn.DropoutWrapper(GRUCell(rnn_dim), state_keep_prob=dropout),
        inputs=word_embeds,
        sequence_length=words_sent,
        dtype=tf.float64)
    words_output = tf.concat((FowardEmbedding, BackwardEmbedding), 2)


# word attention
mask = tf.sequence_mask(words_sent, num_word) # Create Mask True/False
mask = tf.reshape(mask,[-1])

words_u = tf.reshape(words_output,[-1,rnn_dim*2])  # Create hidden units
words_u = tf.layers.dense(words_u, attention_dim, tf.nn.tanh, xavier_init)

words_exp = tf.layers.dense(words_u, 1, tf.exp, False, xavier_init)
words_exp = tf.where(mask, words_exp, tf.zeros_like(words_exp))
words_exp = tf.reshape(words_exp,[-1,num_word,1])
words_exp_sum = tf.reshape(tf.reduce_sum(words_exp,1),[-1,1,1])
word_alpha = words_exp / words_exp_sum

sent_embed = tf.reduce_sum(words_output*word_alpha,1)
sent_embed = tf.expand_dims(sent_embed,0)

# sentence encoding
with tf.variable_scope('sentence'):
    [FowardEmbedding, BackwardEmbedding], _ = tf.nn.bidirectional_dynamic_rnn(
        tf.contrib.rnn.DropoutWrapper(GRUCell(rnn_dim), state_keep_prob=dropout),
        tf.contrib.rnn.DropoutWrapper(GRUCell(rnn_dim), state_keep_prob=dropout),
        inputs=sent_embed,
        sequence_length=tf.expand_dims(num_sent, 0),
        dtype=tf.float64)
    fw = tf.squeeze(FowardEmbedding, [0])
    bw = tf.squeeze(BackwardEmbedding, [0])
    sent_output = tf.concat((fw, bw), 1)


# sentence attention
sent_u = tf.layers.dense(sent_output,attention_dim,tf.nn.tanh, xavier_init)
sent_exp = tf.layers.dense(sent_u, 1, tf.exp, False, xavier_init)
sent_alpha = sent_exp/tf.reduce_sum(sent_exp)
doc_embed = tf.matmul(tf.transpose(sent_alpha),sent_output)


logits = tf.layers.dense(doc_embed, num_classes, kernel_initializer=xavier_init)
prediction = tf.nn.softmax(logits)


def val_score(data, labels):
    correct = 0
    for i in range(len(data)):
        X_input = ConvertDoc2List(data[i])
        if len(X_input) < 1:
            continue
        X_input = ConvertList2Array(X_input)
        feed_dict = {document: X_input, dropout: 1.0}
        val_pred = sess.run(prediction, feed_dict=feed_dict)
        if np.argmax(val_pred) == np.argmax(labels[i]):
            correct +=1
        val_acc = correct/len(data)
    return val_acc


label = tf.placeholder(tf.float64, shape=[num_classes])
labels = tf.expand_dims(label,0)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=labels))
optimizer = tf.train.AdamOptimizer(2e-5,0.9,0.99)
#optimizer   = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
            #save_path = saver.save(sess, savedir + "savedmodels/han_small_50_ep8.ckpt")
            save_path = saver.save(sess, savedir + "savedmodels/han_50_ep8.ckpt")
            #save_path = saver.save(sess, savedir + "savedmodels/han_amazon.ckpt")
        temptime = datetime.timedelta(seconds=round(time.time() - start))
        print("epoch %i, training accuracy: %.2f, validation accuracy: %.2f," % (
        epoch + 1, trainscore * 100, valscore * 100), "time: ", temptime)
        correct = 0
        start = time.time()
totaltime = datetime.timedelta(seconds=round(time.time() - end))
print("\nTime:", totaltime)


# Save the accuracy
accuracy = np.column_stack((np.array(train_acc), np.array(valid_acc)))
#np.savez(datapath + 'accuracy_han_small_50_ep8.npz', accuracy)
np.savez(datapath + 'accuracy_han_50_ep8.npz', accuracy)
#np.savez(datapath + 'accuracy_han_amazon.npz', accuracy)



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

