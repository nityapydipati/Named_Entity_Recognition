#!/bin/python
import os
import string
from collections import defaultdict
import nltk
import re
#import gensim
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
#from nltk.corpus import words as nltkWords
from gensim.models import Word2Vec

#clusters_word=defaultdict(list)
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#model = Word2Vec('google_cluster_model')
#CLUSTERS_NUMBER = 100
#num_features=14
#index2word_set = set(model.index2word)
#model.init_sims(replace=True)

#model.save_word2vec_format('clusters_used.txt', binary=False)    




def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    file_tok = defaultdict(list)
    test=defaultdict(list)
    file_list=[]
    stop_words = []
    file=open("50mpaths2.txt")
    for line in file:
        cluster_info=line.split('\t')
        if len(cluster_info)==3:
            file_tok[cluster_info[1].strip()]=cluster_info[2].strip()
   
    file=open("data/lexicon/tv.tv_program")
    for line in file:
        cluster_info=line.split('\t')
        if len(cluster_info)==2:
            file_tok[cluster_info[0].strip()]=cluster_info[1].strip()

    
        #file_tok[file[13:]]=set(line.strip() for line in open(file))
    
    #for line in open('50mpaths2.txt'):
    #    line=line.strip().split("\t")
        
    #    clusters_word[line[2]].append(line[1])
    return file_tok


def token2features(sent, i, file_tok, tag, stop_words, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    puncts = set(string.punctuation)
    #emoticons_str = r"""
    #(?:
    #    [:=;] # Eyes
    #    [oO\-]? # Nose (optional)
    #    [D\)\]\(\]/\\OpP] # Mouth
    #)"""
     
    
    #tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    #emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
    #Words_List = nltkWords.words()
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    


    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    if word[0].isupper():
        ftrs.append("IS_PRONOUN")
    #if len(word)==0:
    #    ftrs.append("SPACE")
    #if(len(word)>7):
    #    ftrs.append("BIG")
    elif word[0]=="#":
        ftrs.append("HASHTAG")
    elif word[0]=="@":
        ftrs.append("TAGGING_")
    #if '_' in word:
    #    ftrs.append("underline")
    
    #if any(x.isupper() for x in word):
    #    ftrs.append("INNER_UPPER")
    if 'http' in word or '.com' in word:
        ftrs.append("HYPERLINK")
    
    #if word in list(set(puncts)-set(['#','@'])) :
    #    ftrs.append("PUNCT")
    #if emoticon_re.search(word):
    #    ftrs.append("EMOTICON")
    
    for item in file_tok.items():
        if word==item:
            ftrs.append("CLUSTER"+item)


    
    ftrs.append("PREFIX" + word[:3])
    ftrs.append("SUFFIX" + word[-3:])        
    
  
    
    ftrs.append("POSTAG"+tag[i][1])
    ftrs.append("LENGTH"+str(len(word)))
    
    # previous/next word feats
    
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, file_tok, tag, stop_words, add_neighs = False):
                ftrs.append("PREV_" + pf)
        
        if i>1:    
            for pf in token2features(sent, i-2, file_tok, tag, stop_words, add_neighs = False):
                ftrs.append("PREV+1" + pf)
        
        if i>2:
            for pf in token2features(sent, i-3, file_tok, tag, stop_words, add_neighs = False):
                ftrs.append("PREV+2" + pf)
        
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, file_tok, tag, stop_words, add_neighs = False):
                ftrs.append("NEXT_" + pf)
        
        if i<len(sent)-2:
            for pf in token2features(sent, i+2, file_tok, tag, stop_words, add_neighs = False):
                ftrs.append("NEXT+1" + pf)
        
        if i<len(sent)-3:
            for pf in token2features(sent, i+3, file_tok, tag, stop_words, add_neighs = False):
                ftrs.append("NEXT+2" + pf)
        
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    file_tok=preprocess_corpus(sents)
    for sent in sents:
        tag=nltk.pos_tag(sent)
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i, file_tok, tag, stop_words)
