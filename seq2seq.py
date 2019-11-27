#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ananya
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math

import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 80
SOS_token = 0
EOS_token = 1
#****************************************************************************#
#Clean the data by removing punctuations
def TextCleaner(text):
    text=re.sub(r'(\d+)',r'',text)
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u"।",'')
    text=text.replace(u".",'')
    return text
#****************************************************************************#
#Read the file and return the sentences
def GetSentences(text):
    sentences = text.split(u"\n")
    total_sentences = []
    for sent in sentences:
        clean_sen = TextCleaner(sent)
        total_sentences.append(clean_sen)
    return total_sentences
#****************************************************************************#
#return the pair of sentences having
# [reversed words of src sentences, target sentences]
def get_Src_Tgt_Pairs(src,tgt):
    pair = []
    rev_src = []
    #reverse the order of input words 
    for s in range(len(src)):
        words = src[s].split()
        rev_src.append(' '.join(reversed(words)))
        pair.append([rev_src[s],tgt[s]])
    return pair                 
#****************************************************************************#
#to have a unique index per word to use as the inputs & targets.
#Defining a method to track of all word2index and index2word dictionaries
#also keeps track of count of each word word2count
def Vectorize(Dataset):    
    index2word = {0:'SOS', 1:'EOS'}
    cnt = 2 #count SOS, EOS
    word2index = {}
    word2count = {}
    
    for sent in Dataset:
        for word in sent.split(' '):
            if word not in word2index:
                word2index[word] = cnt
                word2count[word] = 1
                index2word[cnt] = word
                cnt+=1
            else:
                word2count[word] +=1
    print('Word count:',cnt)
    return(word2index,index2word,word2count,cnt)

def testData():
    #Dataset Path
    test_hinPath = open('Dataset/test.hi','r',encoding = 'UTF-8')
    test_engPath = open('Dataset/test.en','r',encoding = 'UTF-8')
    #Read the hindi text from file and retrieve the sentences
    test_hinText = test_hinPath.read()
    test_hi_Sentences = GetSentences(test_hinText)
    #Read the english text from file and retrieve the sentences
    test_engText = test_engPath.read()
    test_engText_lower = test_engText.lower()
    test_en_Sentences = GetSentences(test_engText_lower)
    #retrive the source and target pairs (source sentenceare reversed)
    test_pairs = get_Src_Tgt_Pairs(test_en_Sentences,test_hi_Sentences)
    return test_pairs
#****************************************************************************#   
#***************************** TRAINING DATA ********************************#
#Dataset Path
hinPath = open('Dataset/train.hi','r',encoding = 'UTF-8')
engPath = open('Dataset/train.en','r',encoding = 'UTF-8')
#Read the hindi text from file and retrieve the sentences
hinText = hinPath.read()
hi_Sentences = GetSentences(hinText)
#Read the english text from file and retrieve the sentences
engText = engPath.read()
engText_lower = engText.lower()
en_Sentences = GetSentences(engText_lower)
#retrive the source and target pairs (source sentenceare reversed)
pairs = get_Src_Tgt_Pairs(en_Sentences,hi_Sentences)
print(random.choice(pairs))

max_encoder_seq_length = max([len(txt) for txt in en_Sentences])
max_decoder_seq_length = max([len(txt) for txt in hi_Sentences])

#retrieve the vectors (word2index, index2word, word2count) and total word count
en_W2I,en_I2W,en_W2C,en_WrdCnt = Vectorize(en_Sentences)
hi_W2I,hi_I2W,hi_W2C,hi_WrdCnt = Vectorize(hi_Sentences)

#****************************************************************************#
#***********************ENCODER - DECODER NETWORK ***************************#
#****************************************************************************#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 4
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers =4)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 4
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=4)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        


#Prepare Training Data
'''
To train, for each pair we will need:
     "input tensor"  (indexes of the words in the input sentence) 
     "target tensor" (indexes of the words in the target sentence)  '''

def indexesFromSentence(sentence,word2index):
    return [word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(sentence,word2index):
    indexes = indexesFromSentence(sentence,word2index)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0],en_W2I)
    target_tensor = tensorFromSentence(pair[1],hi_W2I)
    return (input_tensor, target_tensor)
    
teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_encoder_seq_length):
    #encoder_hidden = encoder.initHidden()
    h = encoder.initHidden()
    c = encoder.initHidden()
    encoder_hidden = (h,c)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #print(ei,encoder_output[0, 0].shape)
        encoder_outputs[ei] = encoder_output[0, 0]
        
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

#***************************************************************************#
#To print time elapsed and estimated time remaining.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#***************************************************************************#
'''
Start a timer
Initialize optimizers and criterion
Create set of training pairs
Start empty losses array for plotting
'''
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points) 
    
def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        if (iter == 200000 or iter == 400000 or iter == 600000 or iter == 800000):
            encodername = 'encoder'+str(iter)+'.pkl'
            with open(encodername,'wb') as handle:
                pickle.dump(encoder1,handle,protocol = pickle.HIGHEST_PROTOCOL)
            decodername = 'decoder'+str(iter)+'.pkl'
            with open(decodername,'wb') as handle:
                pickle.dump(decoder1,handle,protocol = pickle.HIGHEST_PROTOCOL)
            
    showPlot(plot_losses)
#***************************************************************************#
def evaluate(encoder, decoder, sentence, max_length=max_encoder_seq_length):
    with torch.no_grad():
        input_tensor = torch.tensor([])
        try:
            input_tensor = tensorFromSentence(sentence,en_W2I)
        except KeyError:
            print('')
        input_length = input_tensor.size()[0]
        #encoder_hidden = encoder.initHidden()
        h = encoder.initHidden()
        c = encoder.initHidden()
        encoder_hidden = (h,c)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
       

        for di in range(max_decoder_seq_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(hi_I2W[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(encoder, decoder,pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        
        #print('>', pair[0])
        #pair = ['only surgery through possible is treatment cancer', 'कैंसर का चिकित्सा द्वारा ही सम्भव है ']
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
hidden_size = 256
encoder1 = EncoderRNN(en_WrdCnt, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, hi_WrdCnt).to(device)

'''
Uncomment this code only for training purpose.
Model is already trained and dumped by using pickle.
#trainIters(encoder1, decoder1, 75000, print_every=5000)'''
trainIters(encoder1, decoder1, 1000000, print_every=5000)

# save the model to disk

with open('encoder.pkl','wb') as handle:
    pickle.dump(encoder1,handle,protocol = pickle.HIGHEST_PROTOCOL)
with open('decoder.pkl','wb') as handle:
    pickle.dump(decoder1,handle,protocol = pickle.HIGHEST_PROTOCOL)
'''
Uncomment this when you want to load the previously trained model
encoder2 = pickle.load(open('/home/ananya/Desktop/Code/encoder.pkl',"rb"))
decoder2 = pickle.load(open('/home/ananya/Desktop/Code/encoder.pkl',"rb"))
evaluateRandomly(encoder2, decoder2,pairs)
'''
test_pairs = testData()
evaluateRandomly(encoder1, decoder1,test_pairs)
