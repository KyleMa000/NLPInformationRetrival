# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:33:30 2021

@author: chime
"""
import nltk
import re
import math
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sys


#Read document to list
with open(str(sys.argv[1]), 'r', encoding = "UTF-8") as f:
    test = f.readlines()

doc_key = []
doc_title = []
doc_abstract = []

k = 0
i = 0
while i < len(test):
    if test[i][0:2] == ".I":
        k = test[i][2:].strip()
        doc_key.append(k)
        i+=1

    elif test[i][0:2] == ".W":
        i+=1
        s = ""
        while test[i][0:2] != ".I":
            s = s + test[i][:].replace('\n', ' ')
            i+=1
            if(i == len(test)):
                break
        doc_abstract.append(s)

    elif test[i][0:2] == '.T':
        i+=1
        s = ""
        while test[i][0:2] != ".A":
            s = s + test[i][:].replace('\n', ' ')
            i+=1
        doc_title.append(s)
    else:
        i+=1

doc_query = []
for i in range(len(doc_title)):
    s = doc_abstract[i] + doc_title[i]
    doc_query.append(s)


#read queries to list
with open(str(sys.argv[2]), 'r', encoding = "UTF-8") as f:
    temp = f.readlines()

q_key = []
q_query = []

k = 0
i = 0
while i < len(temp):
    if temp[i][0:2] == ".I":
        k = temp[i][2:].strip()
        q_key.append(k)
        i+=1

    elif temp[i][0:2] == ".W":
        i+=1
        s = ""
        while temp[i][0:2] != ".I":
            s = s + temp[i][:].replace('\n', ' ')
            i+=1
            if(i == len(temp)):
                break
        q_query.append(s)

#Define stop word filter
def filter(query):

    closed_class_stop_words = stopwords.words('english')
    
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    lemmatizer = WordNetLemmatizer()


    for i in range(len(query)):
        query[i] = re.sub(r"\b[0-9]+\b\s*", "", query[i])
        query[i] = lemmatizer.lemmatize(query[i])
        query[i] = tokenizer.tokenize(query[i])
        x =[]
        for j in range(len(query[i])):
            if query[i][j] not in closed_class_stop_words :
                x.append((query[i][j]).lower().strip())
        query[i] = x

#filter stop words from query
filter(q_query)
filter(doc_query)

#recoed word occurances
IDF = {}
for i in range(len(doc_query)):
    for j in range(len(doc_query[i])):
        
        if(doc_query[i][j] not in IDF):
            IDF[doc_query[i][j]] = 0

for element in IDF:
    for i in range(len(doc_query)):
        if element in doc_query[i]:
            IDF[element] +=1



IDF_q = {}

for i in range(len(q_query)):
    for j in range(len(q_query[i])):
        
        if(q_query[i][j] not in IDF_q):
            IDF_q[q_query[i][j]] = 0

for element in IDF_q:
    for i in range(len(q_query)):
        if element in q_query[i]:
            IDF_q[element] +=1



#Calculate IDF, TF and TFIDF scores for each word in queries
q_idf = []
q_tf = []
q_tfidf = []
for i in range(len(q_query)):
    temp_idf = []
    temp_tf = []
    temp_tfidf = []
    for j in range(len(q_query[i])):
        tf = q_query[i].count(q_query[i][j])
        
        tf = tf/len(q_query[i])
        
        temp_tf.append(tf)

        idf = math.log( (len(q_query) / IDF_q[q_query[i][j]] ))
        temp_idf.append(idf)

        temp_tfidf.append(idf * tf)
    q_idf.append(temp_idf)
    q_tf.append(temp_tf)
    q_tfidf.append(temp_tfidf)
    
q_vector_word = []
for i in range(len(q_query)):
    for j in range(len(q_query[i])):
        q_vector_word.append(q_query[i][j])

#Calculate IDF, TF and TFIDF scores for each word in abstracts
doc_idf = []
doc_tf = []
doc_tfidf = []
for i in range(len(doc_query)):
    temp_idf = []
    temp_tf = []
    temp_tfidf = []
    for j in range(len(doc_query[i])):
        tf = doc_query[i].count(doc_query[i][j])
        
        tf = tf/len(doc_query[i])
        
        
        temp_tf.append(tf)
        
        idf = math.log( (len(doc_query)/IDF[doc_query[i][j]]))
        temp_idf.append(idf)

        temp_tfidf.append(idf * tf)
    doc_idf.append(temp_idf)
    doc_tf.append(temp_tf)
    doc_tfidf.append(temp_tfidf)

cos = []
for i in range(len(q_query)):
    cos_score = []
    for j in range(len(doc_query)):
        vector = []
        for x in range((len(q_query[i]))):
            if q_query[i][x] in doc_query[j]:
                a = doc_query[j].index(q_query[i][x])
                vector.append(doc_tfidf[j][a])
            else:
                vector.append(0)
        
        top = 0
        b1 = 0
        b2 = 0
        for d in range(len(vector)):
            top = top + (vector[d] * q_tfidf[i][d])
            b1 = b1 + (vector[d] * vector[d])
            b2= b2 + (q_tfidf[i][d] * q_tfidf[i][d])
        try:
            similarity = top / math.sqrt(b1 * b2)
        except:
            similarity = 0
            
        cos_score.append([j+1, similarity])
    
    df = pd.DataFrame(cos_score, columns = ["index", "similarity"])
    df = df.sort_values(by = ["similarity"], ascending=False)
    df = df[df["similarity"] != 0]
    df = df.values.tolist()[0:100]
    cos.append(df)



f = open('output.txt', 'w', encoding = "UTF-8", newline='\n')

for i in range(len(cos)):
    file = str(i+1)
    for j in range(len(cos[i])):
        num = str(int(cos[i][j][0]))
        sco = str(round(cos[i][j][1], 3))
        f.write(file + ' ' + num +' ' + sco + '\n')
f.close()


