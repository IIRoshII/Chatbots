# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:29:40 2018

@author: USER 1
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer=LancasterStemmer()

import numpy as np
import random
import json 
with open('intents.json') as json_data:
    intents=json.load(json_data)
    
words=[]
classes=[]
documents=[]
ignore_words=['?','-']

for  intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))

classes=sorted(list(set(classes)))

print(len(documents),"documents")
print(len(classes),"classes",classes)
print(len(words),"unique stemmed words",words)

training=[]
output=[]
output_empty=[0]*len(classes) #empty array with the length of number of intents

for doc in documents:#doc represents each sentence in the document(along with the tag)
    bag=[] #each bag is created for each element in the document
    pattern_words=doc[0] #doc[0]=sentence doc[1] is tag. for each sentence pattern_word is created
    pattern_words=[stemmer.stem(word.lower()) for word in pattern_words] #stemming 
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) # for the sentence the bag marks 1 if it is available in the stemmed words else 0
        
        
    output_row=list(output_empty) #output row is a list
    output_row[classes.index(doc[1])]=1 #particular tag is marked 1

    training.append([bag,output_row])# 58 (bag of 1 and 0) + 7 (intent tag of 1 and 0)
 
random.shuffle(training)

training=np.array(training) #converting from list to array

train_x=list(training[:,0])# bag col
train_y=list(training[:,1])# intent col
