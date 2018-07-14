#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Larix'

import jieba
import math
import os
import json

from collections import OrderedDict

class TF_IDF():
    def __init__(self):
        self.docs = {}
        self.seg_docs = self.get_seg_docs()
        self.stopword = []
        self.tf = []
        self.df = {}
        self.idf = {}
        self.topK_idf = {}
        self.bow = {}
        self.cal_tfidf()

    def read_file(self, path, type):
        if type == 'json':
            with open(path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
        elif type == 'txt':
            with open(path, 'r', encoding='utf-8') as file:
	            data = file.read()
        return data

    def get_seg_docs(self):
        _seg_docs = []
        FOLDER_NAME = 'data'    
        DOCUMENT = 'news_data.json'
        STOPWORD = 'stopword.txt'
        FILE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], FOLDER_NAME)

        self.docs = self.read_file(FILE_DIR + '\\' + DOCUMENT, 'json')
        self.stopword = self.read_file(FILE_DIR + '\\' + STOPWORD, 'txt')
        for i in range(len(self.docs)):
            content_seg = [w for w in jieba.lcut(self.docs[i]['content']) if len(w) > 1 and w not in self.stopword and w.isalpha()]
            _seg_docs.append(content_seg)
        return _seg_docs
    """
    計算tf,idf結果
    tf:[{word1:3,word2:4,word4:2},{word2:5,word3:7, word4:2},{....},.......]
    df:{word1:6個文檔,word2:3個文檔,word3:5個文檔,word4:4個文檔......}
    idf:{word1:idf(word1),word2:idf(word2),word3:idf(word3)..........}
    """
    def cal_tfidf(self):
        for doc in self.seg_docs:
            bow = {}
            for word in doc:
                if not word in bow:
                    bow[word] = 0
                bow[word] += 1
            self.tf.append(bow)
            for word, _ in bow.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1
        for word, df in self.df.items():
            #只出現過在一篇文檔的詞不要(選擇性)
            if df < 2:
                pass
            else:
                self.idf[word] = math.log10(len(self.seg_docs) / df)

    def tf(self, index, word):
    	return self.tf[index][word]

    def idf(self, word):
    	return self.idf[word]

    def tf_idf(self, index, word):
        return self.tf[index][word]*self.idf[word]

    def get_topK_idf(self, k, reverse = True):
        self.topK_idf = OrderedDict(sorted(self.idf.items(), key=lambda t: t[1], reverse = reverse)[:k])
        return  self.topK_idf

    def get_docment(self):
        return self.docs

    def get_title(self, index):
        return self.docs[index]['title']

    def get_content(self, index):
        return self.docs[index]['content']

    def set_bag_of_word(self, bow):
        self.bow = bow

    def get_text_vector(self, index):
        return  [1*self.tf_idf(index, w) if w in jieba.lcut(self.docs[index]['content']) else 0 for w in self.bow]

    def cosine_similarity(self, v1, v2):
	    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
	    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
	    for i in range(0, len(v1)):
	    	x, y = v1[i], v2[i]
	    	sum_xx += math.pow(x, 2)
	    	sum_yy += math.pow(y, 2)
	    	sum_xy += x * y
	    try:
	        return sum_xy / math.sqrt(sum_xx * sum_yy)	
	    except ZeroDivisionError:
	        return 0
def main():
    tf_idf = TF_IDF()
    topK = tf_idf.get_topK_idf(1000, True)
    #保存bag of word
    tf_idf.set_bag_of_word(set(topK.keys()))
    #得到文章第1篇跟第11篇的向量
    vec1 = tf_idf.get_text_vector(0)
    vec2 = tf_idf.get_text_vector(10)
    #計算文件與文件的cosine similarity
    score1 = tf_idf.cosine_similarity(vec1, vec1)
    score2 = tf_idf.cosine_similarity(vec1, vec2)

    print(topK, score1, score2)


if __name__ == '__main__':
    main()
