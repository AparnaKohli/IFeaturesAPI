import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
  
def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN


def perform_Stemming(line):
    if (line.startswith('[t]') or line.startswith('*')):
        return line
    review_with_aspect=line.replace('(','').replace(')','').replace('-','').split('##')
    Header=review_with_aspect[0]
    sentence=review_with_aspect[1]    
    postags = pos_tag(word_tokenize(sentence))
    lemmatizer = WordNetLemmatizer()
    stemmedSentence=""
    HeaderContextWordList=[]

    for item in Header.split(','):
        if len(item.split('@'))>1:
            for xitem in item.split('@')[1].split(':'):
                for yitem in xitem.split():
                    HeaderContextWordList.append(yitem.replace(']',''))

    #print(HeaderContextWordList)
    
    for item in postags:
        stemmedSentence+=lemmatizer.lemmatize(item[0],penn_to_wn(item[1]))+" "
        if item[0] in HeaderContextWordList:
            Header=Header.replace(item[0]+':',lemmatizer.lemmatize(item[0],penn_to_wn(item[1]))+':')
            Header=Header.replace(item[0]+']',lemmatizer.lemmatize(item[0],penn_to_wn(item[1]))+']')
            #print((item[0],lemmatizer.lemmatize(item[0],penn_to_wn(item[1]))))
    return Header+"##"+stemmedSentence
