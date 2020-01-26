
#import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.modules.module import Module
from torch.utils.data import TensorDataset

import math
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import time
import datetime
import string
#from tensorflow.contrib.layers import layer_norm
from nltk import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split

datapath = 'F:/UChicago/degree paper/Archive/' # Datapath where the data is saved
savedir  = 'F:/UChicago/degree paper/Archive/save/'      # Savepath to save the result from the model


# Load sample dataset
#npz = np.load(datapath + 'yelp_review_small.npz', allow_pickle=True)
#npz = np.load(datapath + 'yelp_review_1000.npz', allow_pickle=True)
#npz = np.load(datapath + 'yelp_review_new.npz', allow_pickle=True)
#npz = np.load(datapath + 'yelp_review.npz', allow_pickle=True)
#npz = np.load(datapath + 'amazon_review_100k.npz', allow_pickle=True) #Tool dataset
#npz = np.load(datapath + 'amazon_review_100k_cell.npz', allow_pickle=True) #cell dataset
npz = np.load(datapath + 'amazon_review_100k_pet.npz', allow_pickle=True) #pet dataset
data = npz['arr_0']
data[:3,]

X = data[:,0]  # Text
Y = data[:,1]  # Label
del data

## Set the seed of the model
seed=14
torch.manual_seed(seed)
np.random.seed(seed)

# Transfer Label to Onehot encoding
le = LabelEncoder()
y = le.fit_transform(Y)
lb = LabelBinarizer()
lb.fit(y)
LabelEncoding = lb.transform(y)

## Set the data set, split the data into train, valid, and test data set
# (Train, Valid, Test)=(0.8, 0.1, 0.1)
X_train,X_test,y_train,y_test = train_test_split(X,LabelEncoding,
                                                 test_size=0.2,random_state=seed,stratify=LabelEncoding)
X_valid,X_test,y_valid,y_test = train_test_split(X_test,y_test,
                                                test_size=0.5,random_state=seed,stratify=y_test)

## Create Model
## Relevant functions
# Precomputed word2idx dictionary
#with open('%sword2idx_yelp_small.json' % datapath) as f:
#with open('%sword2idx_yelp_1000.json' % datapath) as f:
#with open('%sword2idx_yelp_new.json' % datapath) as f:
#with open('%sword2idx_yelp.json' % datapath) as f:
#with open('%sword2idx_amazon_100k.json' % datapath) as f:
#with open('%sword2idx_amazon_100k_cell.json' % datapath) as f:
with open('%sword2idx_amazon_100k_pet.json' % datapath) as f:
    w2i = json.load(f)

# Precomputed wordembedding matrix with dimension 50
#npz = np.load('%sWordEmbedding_yelp_small.npz' % datapath)
#npz = np.load('%sWordEmbedding_yelp_1000.npz' % datapath)
#npz = np.load('%sWordEmbedding_yelp_new.npz' % datapath)
#npz = np.load('%sWordEmbedding_yelp.npz' % datapath)
#npz = np.load('%sWordEmbedding_amazon_100k.npz' % datapath)
#npz = np.load('%sWordEmbedding_amazon_100k_cell.npz' % datapath)
npz = np.load('%sWordEmbedding_amazon_100k_pet.npz' % datapath)
WordEmbeddings = npz['arr_0']


# Remove puntuation
def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation+"\n"))

# Convert sentence to word
def ConvertSentence2Word(s):
    return(word_tokenize(remove_punctuation(s).lower()))

# Convert sentence to word idx
def ConvertSent2Idx(s):
    s_temp = [w for w in ConvertSentence2Word(s) if w in w2i]
    temp = []
    for w in s_temp:
        temp.append(w2i[w])
    return(temp)

# Divide Document with variable length of sentences
def ConvertDoc2List(doc):
    temp_doc = sent_tokenize(doc)
    temp = []
    for i in range(len(temp_doc)):
        if(len(ConvertSent2Idx(temp_doc[i]))>=1):      # Prevent empty sentence
            temp.append(ConvertSent2Idx(temp_doc[i]))
    return(temp)

# Convert List type of document to array
def ConvertList2Array(docs):
    ms = len(docs)
    mw = len(max(docs, key=len))
    result = np.zeros((ms,mw))
    for i, line in enumerate(docs):
        for j, word in enumerate(line):
            result[i,j] = word
    return result



###############################################################################################################

## Define the class of self attention
class ConvolutionalSelfAttention(Module):
    def __init__(self, input_dim, kernel_dim, dropout_rate, conv_cnt=1): #convc_cnt=3/1
        super(ConvolutionalSelfAttention, self).__init__()
        self.input_dim = input_dim
        ## initialize the convolution layer. conv_cnt=3 for initialization of K, Q, V
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, input_dim, kernel_dim, padding=1)
                                    for _ in range(conv_cnt)])

        for w in self.convs:
            #nn.init.xavier_normal_(w.weight)
            nn.init.kaiming_uniform_(w.weight)  #HE initialization
        self.dropout = nn.Dropout(1-dropout_rate)
        self.cls = nn.Linear(input_dim, input_dim) # similar to tf.layers.dense
        nn.init.xavier_normal_(self.cls.weight) #Xavier initialization

    ## define the attention process: softmax(Q*K^T/sqrt(d))*V
#    def attention(self, q, k, v):
        #return torch.softmax(torch.div(torch.bmm(q.permute(0,2,1), k),
        #        np.sqrt(self.input_dim)), 2).bmm(v.permute(0,2,1)).permute(0,2,1)
#        return self.dropout(torch.softmax(torch.div(torch.bmm(q.permute(0,2,1), k),
#                 np.sqrt(self.input_dim)), 2)).bmm(v.permute(0,2,1)).permute(0,2,1)
        #return torch.softmax(torch.div(torch.bmm(q, k.permute(0, 2, 1)),
        #         np.sqrt(self.input_dim)), 2).bmm(v)

    def attention(self,V,input):
        #return [F.relu(conv(V)) for conv in self.convs]
        return [F.relu(conv(V)+input) for conv in self.convs]


    def forward(self, input):
        # initialize Q, K, V
        hiddens = [F.relu(conv(input)) for conv in self.convs]

        #compute the relu_hid, which is the results of attention process.
#        relu_hid = self.attention(hiddens[0], hiddens[1], hiddens[2])
        relu_hid = self.attention(hiddens[0],input)[0]

        # If only use the Norm layer
#        output = F.layer_norm(relu_hid, relu_hid.size()[1:])  # Norm layer
#        print(relu_hid.size()[1:])
        #output = F.layer_norm(relu_hid+input, relu_hid.size()[1:])  # Norm layer
        output = F.layer_norm(relu_hid, relu_hid.size()[1:])  # Norm layer
        ## add an add in the norm

        #relu_hid = F.layer_norm(relu_hid+input, relu_hid.size()[1:]) #Add and Norm layer
        #encoder = self.cls(relu_hid.permute(0, 2, 1)).permute(0, 2, 1) #Feed Forward layer
        #output = F.layer_norm(relu_hid+encoder, relu_hid.size()[1:]) #Add and Norm layer

        # Then we can get the convolutional Attention

        ##output = F.layer_norm(relu_hid + input, relu_hid.size()[1:])

        return output

# Define the class of Target Attention.
class TargetAttention(ConvolutionalSelfAttention):
    def __init__(self, input_dim, kernel_dim, dropout_rate, conv_cnt=2):
        # conc_cnt=2 for initialize of K and V
        super(TargetAttention, self).__init__(input_dim, kernel_dim, dropout_rate, conv_cnt)
        # define the transfer embedding vector T\in R^{1*d}
        #self.target = nn.Parameter(torch.randn(input_dim, 1))
        self.target = nn.Parameter(torch.empty((input_dim, 1)))
        nn.init.kaiming_uniform_(self.target)
        #stdv = 1. / math.sqrt(self.target.size(1))
        #self.target.data.uniform_(-stdv,stdv)

    #define the target attention process: softmax(T*K^T/sqrt(d))*V
    def target_att(self, t, k, v):
        return self.dropout(torch.softmax(torch.div(torch.bmm(t.permute(0, 2, 1), k),
                np.sqrt(self.input_dim)), 2)).bmm(v.permute(0, 2, 1)).permute(0, 2, 1)
        #return torch.softmax(torch.div(torch.bmm(t.permute(0, 2, 1), k),
        #        np.sqrt(self.input_dim)), 2).bmm(v.permute(0, 2, 1)).permute(0, 2, 1)
        #return torch.softmax(torch.div(torch.bmm(t, k.permute(0, 2, 1)),
        #            np.sqrt(self.input_dim)), 2).bmm(v)

    def forward(self, input):
        batch_size = input.size(0)
        K = input
        V = input
        #hiddens = [F.relu(conv(input)) for conv in self.convs] #initialize K, V
        # compute the restuls of target attention
    #    relu_hid = self.target_att(self.target.expand(batch_size, self.input_dim, 1), hiddens[0], hiddens[1])
        relu_hid = self.target_att(self.target.expand(batch_size, self.input_dim, 1), K, V)
        output = relu_hid
        #output = self.AddNorm([self.target.expand(batch_size, self.input_dim, 1)]+hiddens)
        return output

class Hierarchical(nn.Module):
    def __init__(self, num_emb, input_dim, dropout_rate, pretrained_weight1, pretrained_weight2):
        super(Hierarchical, self).__init__()
        #self.id2vec = nn.Embedding(num_emb, input_dim, padding_idx=1)
        # Define the initialization of embedding matrix
        # One is using the pretrained weight matrix, the other is initialized randomly.
        self.id2vec1 = nn.Embedding(num_emb, 50, padding_idx=1)
        self.id2vec1.weight.data.copy_(pretrained_weight1)
        self.id2vec1.requires_grad=True
        self.id2vec2 = nn.Embedding(num_emb, 50, padding_idx=1)
        self.id2vec2.weight.data.copy_(pretrained_weight2)
        self.id2vec2.requires_grad = True
        self.dropout = nn.Dropout(1-dropout_rate)

        self.loss=nn.CrossEntropyLoss()

    def binary_accuracy(self,y_pred,y):
        ge = torch.ge(y_pred.type(y.type()),0.5).float()
        correct = torch.eq(ge,y).view(-1)
        return torch.sum(correct).item()/correct.shape[0]

    def accuracy(self,y_pred,y):
        # compute the number of correct of the prediction.
        _, pred=y_pred.max(1)
        return sum(pred==y).cpu().numpy()/y.size()

class HMCAN(Hierarchical):
    def __init__(self, input_dim, hidden_dim, kernel_dim,
                dropout_rate, num_emb, pretrained_weight1, pretrained_weight2):
        super(HMCAN, self).__init__(num_emb, input_dim, dropout_rate, pretrained_weight1, pretrained_weight2)

        #self.positions = nn.Parameter(torch.randn(sent_maxlen, input_dim))
        #stdv = 1. / self.positions.size(1) ** 0.5
        #self.positions.data.uniform_(-stdv,stdv)
        self.csa = ConvolutionalSelfAttention(input_dim, kernel_dim, dropout_rate)
        self.ta = TargetAttention(input_dim, kernel_dim, dropout_rate)
        self.dropout = nn.Dropout(1-dropout_rate)
        self.cls = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_normal_(self.cls.weight)

    def predict(self, x):
        # for the input data, compute the embedding matrix (using multichannel)
        input1 = self.id2vec1(x)
        input2 = self.id2vec2(x)
        input = self.dropout(input1 + input2)
        #input = self.dropout(input+self.positions)

        #print(input.shape)

        time1 = time.time() # get record of the time

        # Word Hierarchy, with convolutional attention and target attention
        # Using Word Embedding, dim = l*d, d=50
        hidden = self.csa(input.permute(0,2,1))
        hidden = self.ta(hidden)
        # Output dim: 1*d

        time2 = time.time() # get time

        hidden = hidden.permute(2, 0, 1) #Change the dim of the hidden data to fit the model

        #print(hidden.shape)

        # Sentence Hierarchy. Use the Sentence embedding, dim = 1*d, d=50
        hidden = self.csa(hidden.permute(0,2,1))
        hidden = self.ta(hidden)
        # Output dim: 1*50

        time3 = time.time()

        # To compute the Document Embedding, dim = 1*5
        logits = self.cls(hidden.squeeze(-1))

        # Compute the time of each part
        t1 = time2 - time1
        t2 = time3 - time2
        t3 = time.time() - time3

        return logits, t1, t2, t3

    def forward(self, x,  y):
        logits, t1, t2, t3 = self.predict(x)
        ## drop out the logits here
        label = torch.Tensor([np.argmax(y.cpu()).squeeze()])
        #print(logits)
        #print(label)
        loss = self.loss(logits, label.long().cuda()) #When using GPU, we use .cuda()
        accuracy = self.accuracy(logits, label.long().cuda())

        return loss, accuracy, t1, t2, t3

def val_score(model, data, labels):
    #Test the model accuracy for the test data set (or valid data set)
    correct = 0
    #l = labels.shape[1]
    for i in range(len(data)):
        X_input = ConvertDoc2List(data[i])
        if len(X_input) < 1:
            continue
        X_input = torch.LongTensor(ConvertList2Array(X_input))
        X_input = X_input.cuda()
        val_pred, _, _, _ = model.predict(X_input)

        #torch.Tensor([np.argmax(y).squeeze()])

        label = torch.LongTensor([np.argmax(labels[i]).squeeze()])
        acc = model.accuracy(val_pred, label.cuda())
        correct = correct + acc # Compute the total correct number

        #if np.argmax(val_pred) == np.argmax(labels[i]):
        #    correct +=1
    val_acc = correct/len(data)

    return val_acc

### input_dim   dim of input data, number of words
### num_emb: dim of the embedding matrix  wordEmbeddings.shape[0]
### pretrained_weight: embedding matrix
### hidden_dim: num of classes
### kernel_dim: kernel dim = 3
### sent_maxlen: max length of sentences
### dropout_rate: dropout rate
### about multichannel, two different pretrained weight, one is WordEmbeddings, the other is initialized by xavier_init
### torch.nn.init.kaiming_uniform(tensor, a=0, mode='fan_in') HE initializer

# Set the parameters
input_dim = WordEmbeddings.shape[1] # = 50
kernel_dim = 3  #
num_emb = WordEmbeddings.shape[0] # number of words in the embedding matrix
dropout_rate = 0.9 # 1-dropout_rate=0.1
pretrained_weight1 = torch.Tensor(WordEmbeddings) # Word Embedding matrix 1
pretrained_weight2 = torch.Tensor(WordEmbeddings.shape[0],WordEmbeddings.shape[1])
pretrained_weight2 = nn.init.xavier_normal_(pretrained_weight2) # Word Embedding matrix 2
hidden_dim = LabelEncoding.shape[1] # = 5

n_epochs = 30 # When we remove the dropout layer and softmax layer, it converges fast.
start = time.time()
end = time.time()
#torch.manual_seed(seed=15)
# Define the model
model = HMCAN(input_dim, hidden_dim, kernel_dim,
            dropout_rate, num_emb, pretrained_weight1, pretrained_weight2)
#optimizer=torch.optim.Adam(model.parameters(), eps=2e-5,betas=(0.9,0.99)) # define the optimizer function
#optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001, rho=0.5, eps=2e-5, weight_decay=0)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

# If we use gpu to run the model.
gpus = [0]   #use which gpu
cuda_gpu = torch.cuda.is_available()   #Check if GPU is available
if(cuda_gpu):
    model = model.cuda()   #change the model type to cuda

### if the cuda() is the same thing as .to(device)

train_acc = []
valid_acc = []
bestscore = 0
accuracy = 0

X_input_data = []
y_input_data = []
# First we will process the data, transfer them to array
for i in range(len(X_train)):
    X_input = ConvertDoc2List(X_train[i])
    if(len(X_input)<1):
        continue
    X_input = torch.LongTensor(ConvertList2Array(X_input))
    y_input = torch.LongTensor(y_train[i])
    X_input_data.append(X_input)
    y_input_data.append(y_input)

# Then we run the model in each epochs.
for epoch in range(n_epochs):
    time1 = []
    time2 = []
    time3 = []
    time4 = []
    for i in range(len(X_input_data)):
        # get the input data
        X_input = X_input_data[i]
        y_input = y_input_data[i]

        #X_input = ConvertDoc2List(X_train[i])
        #if(len(X_input)<1):
        #    continue
        #X_input = torch.LongTensor(ConvertList2Array(X_input))
        #y_input = torch.LongTensor(y_train[i])

        if cuda_gpu:
            # if use gpu, use .cuda()
            X_input = X_input.cuda()
            y_input = y_input.cuda()

        # train the model with input data
        optimizer.zero_grad()
        loss, cort, t1, t2, t3 = model.forward(x=X_input,y=y_input)
        tt = time.time()
        loss.backward()
        optimizer.step()
        accuracy = accuracy + cort

        t4 = time.time() - tt
        time1.append(t1)
        time2.append(t2)
        time3.append(t3)
        time4.append(t4)
    accuracy = accuracy / len(X_train)
    #print(accuracy)
    train_acc.append(accuracy)
    tt = time.time()
    # test the model on the valid set
    model = model.eval()
    valscore = val_score(model,X_valid,y_valid)
    model = model.train()
    valid_acc.append(valscore)
    timetest = time.time() - tt

    #print(np.sum(time1),np.sum(time2),np.sum(time3),np.sum(time4), timetest)
    ## time1: time of word hierarchy; time2: time of sentence hierarchy; time3: time of classify
    ## time4: time of update the model
    # save the best model
    if valscore >= bestscore:
        bestscore = valscore
        #save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_small_50.pkl")
        #save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_1000.pkl")
        #save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_small_new.pkl")
        #save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_50.pkl")
        #save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_amazon.pkl")
        #save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_amazon_cell.pkl")
        save_path = torch.save(model, savedir + "savedmodels/hmmcan_py_amazon_pet.pkl")
    temptime = datetime.timedelta(seconds=round(time.time() - start))
    print("epoch %i, training accuracy: %.2f, validation accuracy: %.2f," % (
        epoch + 1, accuracy * 100, valscore * 100), "time: ", temptime)
    accuracy = 0
    start = time.time()
totaltime = datetime.timedelta(seconds=round(time.time() - end))
print("\nTime:", totaltime)

# Save Accuracy
accuracy = np.column_stack((np.array(train_acc), np.array(valid_acc)))
#np.savez(datapath + 'accuracy_hmmcan_py_small_50.npz', accuracy )
#np.savez(datapath + 'accuracy_hmmcan_py_1000.npz', accuracy )
#np.savez(datapath + 'accuracy_hmmcan_py_small_new.npz', accuracy )
#np.savez(datapath + 'accuracy_hmmcan_py_50.npz', accuracy )
#np.savez(datapath + 'accuracy_hmmcan_py_amazon.npz', accuracy )
#np.savez(datapath + 'accuracy_hmmcan_py_amazon_cell.npz', accuracy )
np.savez(datapath + 'accuracy_hmmcan_py_amazon_pet.npz', accuracy )

# Plot the accuracy
epochs = range(1,n_epochs+1)
plt.plot(epochs, train_acc, label="train")
plt.plot(epochs, valid_acc, label="valid")
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# test the model on the Test data
start = time.time()
#model = torch.load(savedir + "savedmodels/hmmcan_py_small_50.pkl")
#model = torch.load(savedir + "savedmodels/hmmcan_py_1000.pkl")
#model = torch.load(savedir + "savedmodels/hmmcan_py_small_new.pkl")
#model = torch.load(savedir + "savedmodels/hmmcan_py_50.pkl")
#model = torch.load(savedir + "savedmodels/hmmcan_py_amazon.pkl") ## Tool Dataset
#model = torch.load(savedir + "savedmodels/hmmcan_py_amazon_cell.pkl") ## Cell dataset
model = torch.load(savedir + "savedmodels/hmmcan_py_amazon_pet.pkl") ## Pet dataset
model = model.eval()
testscore = val_score(model, X_test, y_test)
totaltime = datetime.timedelta(seconds=round(time.time()-start))
print("\ntest accuracy: %.2f" % (testscore*100),"%")
print("Time:", totaltime)

test_acc_hcan = testscore





