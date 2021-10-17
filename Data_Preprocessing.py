# -*- coding: utf-8 -*-
# encoding=utf8
from __future__ import absolute_import
from pattern.en import suggest
from collections import Counter
import tweepy
import os
import sys
import string
import re
import pickle
import preprocessor as p
import numpy as np
import nltk
from collections import Counter
from nltk import tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
#from html.parser import HTMLParser
import html
import unicodedata2
import itertools
from nltk.corpus import wordnet
from textblob import TextBlob

p.set_options(p.OPT.MENTION)
Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)


#raw_tweet="RT @shanu i Well i myself his @CB_Baby24: @white_thunduh I'd love &amp; to be :) :) ? ! | @ < > []~     sleeping right now... YEAH But nooooooo lets just ashraf's stay are..it awake all might. That's what I wanted to do.. https://t.co/6kUsbZgsnD .#sarcasm. #cantsleep. #blah. "
#print (raw_tweet)

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'these', 'your', 'his', 'through', 'don', 'nor', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'their', 'above', 'both', 'up', 'to', 'had', 'she', 'all', 'no', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'which', 'those', 'after', 'few', 'whom', 'if', 'theirs', 'against', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than',  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'im','?','!','!!','!!!','!!!!']

def Stop_Words(text):
    try:
        english_words=[]
        word_tokens = word_tokenize(text)
        word_tokens = [word for word in word_tokens if word.isalpha()]
        filtered_text = [w for w in word_tokens if not w in stop_words]
        text_final=' '.join(map(str, filtered_text))
        return text_final
    except:
        print('An error occurred in Stop_Words.')
    

# This method is for extracting contraction.
def Expand_Contractions(text):
    text = text.replace(" can't ", ' cannot ').replace("can't've ", ' cannot have ').replace("'cause ", ' because ').replace("ain't ", ' am not ')\
        .replace("could've ", ' could have ').replace("couldn't ", ' could not ').replace("couldn't've ", ' could not have ')\
        .replace("doesn't ", ' does not ').replace("don't ", ' do not ').replace("hadn't ", ' had not ').replace("hadn't've ", ' had not have ') \
        .replace("hasn't ", ' has not ').replace("haven't ", ' have not ').replace("he'd ", ' he would ').replace("he'd've ", ' he would have ') \
        .replace("he'll ", ' he will ').replace("he'll've ", ' he will have ').replace("he's ", ' he is ').replace("how'd ", ' how did ') \
        .replace("how'd'y ", ' how do you ').replace("how'll ", ' how will ').replace("how's ", ' how is ').replace("I'd ", ' I would ') \
        .replace("I'd've ", ' I would have ').replace("I'll ", ' I will ').replace("I'll've ", ' I will have ').replace("I'm ", ' I am ') \
        .replace("I've ", ' I have ').replace("isn't ", ' is not ').replace("it'd ", ' it had ').replace("it'd've ", ' it would have ') \
        .replace("it'll ", ' it will ').replace("it'll've ", ' it will have ').replace("it's ", ' it is ').replace("let's ", ' let us ') \
        .replace("ma'am ", ' madam ').replace("mayn't ", ' may not ').replace("might've ", ' might have ').replace("mightn't ", ' might not ') \
        .replace("mightn't've ", ' might not have ').replace("must've ", ' must have ').replace("might've ", ' might have ') \
        .replace("mustn't've ", ' must not have ').replace("needn't ", ' need not ').replace("needn't've ", ' need not have ') \
        .replace("oughtn't", ' ought not ').replace("oughtn't've ", ' ought not have').replace("shan't ", ' shall not ')\
        .replace("shan't've", ' shall not have ').replace("sha'n't've ", ' shall not have').replace("she'd ", ' she would ')\
        .replace("mustn't ", ' must not ').replace("aren't ", ' are not ').replace("o'clock ", ' of the clock ').replace("sha'n't ", ' shall not ') \
        .replace("she'd've ", ' she would have ').replace("she'd've ", ' she would have ').replace("o'clock ", ' of the clock ') \
        .replace("sha'n't ", ' shall not ').replace("she'll ", ' she will ').replace("she'll've ", ' she will have ').replace("she's ", ' she is ')\
        .replace("should've ", ' should have ').replace("shouldn't ", ' should not ').replace("shouldn't've ", ' should not have ')\
        .replace("so've ", ' so have ').replace("didn't ", ' did not ').replace("so's ", ' so is ').replace("that'd ", ' that would ')\
        .replace("that'd've ", ' that would have ').replace("that's ", ' that is ').replace("there'd ", ' there had ').replace("there's ", ' there is ')\
        .replace("there'd've ", ' there would have ').replace("they'd ", ' they would ').replace("they'd've ", ' they would have ')\
        .replace("they'll ", ' they will ').replace("they'll've ", ' they will have ').replace("they're ", ' they are ').replace("they've ", ' they have ')\
        .replace("to've ", ' to have ').replace("wasn't ", ' was not ').replace("we'd ", ' we had ').replace("we'd've ", ' we would have ')\
        .replace("we'll ", ' we will ').replace("we'll've ", ' we will have ').replace("we're ", ' we are ').replace("we've ", ' we have ')\
        .replace("weren't ", ' were not ').replace("what'll ", ' what will ').replace("what'll've ", ' what will have').replace("what're ", ' what are ')\
        .replace("what's ", ' what is ').replace("what've ", ' what have ').replace("when's ", ' when is').replace("when've ", ' when have ')\
        .replace("where'd ", ' where did ').replace("where's ", ' where is ').replace("where've ", ' where have').replace("who'll ", ' who will ')\
        .replace("who'll've ", ' who will have ').replace("who's ", ' who is ').replace("who've ", ' who have').replace("why's ", ' why is ')\
        .replace("why've ", ' why have ').replace("will've ", ' will have ').replace("won't ", ' will not ').replace("won't've ", ' will not have ')\
        .replace("would've ", ' would have ').replace("wouldn't ", ' would not ').replace("wouldn't've ", ' would not have').replace("'s ",' is ')\
        .replace("y'all ", ' you all ').replace("y'alls ", ' you alls ').replace("y'all'd ", ' you all would').replace("y'all'd've ", ' you all would have ')\
        .replace("y'all're ", ' you all are ').replace("y'all've ", ' you all have ').replace("you'd ", ' you had').replace("you'd've ", ' you would have ')\
        .replace("you'll ", ' you will ').replace("you'll've ", ' you will have ').replace("you're ", ' you are').replace("you've ", ' you have')\
        .replace("cant ", ' cannot').replace("i'm", ' i am').replace("im", ' i am').replace("can t ", ' cannot').replace("mayt ", ' maynot')    
    return text;

# This method is for removing URL.
def Strip_Links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ' ')    
    return text

# Twitter text comes HTML-escaped, so unescape it.
# We also first unescape &amp;'s, in case the text has been buggily double-escaped.
def Remove_Ampersand(text):
    #text = text.replace("&amp;", "&")
    text = text.replace("|", "")
    text = text.replace("?", "")
    text = text.replace("?", "")
    text = text.replace("!", "")  
    text=html.unescape(text)   
    #text = HTMLParser.HTMLParser().unescape(text)
    return text

# "am   I  " => "am I"
# Remove White Spaces
def Remove_Extra_Whitespace(text):
    #return re.sub( '\s+', ' ', input).strip()
    #return Whitespace.sub(" ", input).strip()
    s=' '.join(text.split())
    return s

# This function is to convert lower case letter
def Convert_Lowercase(preprocessed_text):
    return preprocessed_text.lower()

# This function is to convert upper case letter
def Convert_Uppercase(preprocessed_text):
    return preprocessed_text.upper()    

# This function is to convert Camel case letter
def Convert_Camelcase(preprocessed_text):
    return preprocessed_text.title()

# This function is to remove hex care
def Remove_Hexchar(preprocessed_text):
    return preprocessed_text.encode('ascii', errors='ignore')

# This function is to remove @user in text.
def Remove_Retweets(text):
	return re.sub('@[^\s]+','AT_USER',text)
        
def ReplaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    #preprocessed_tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(preprocessed_tweet))
    return  pattern.sub(r"\1\1", s.decode('utf-8'))
    
def Remove_charaters(text):
    #Convert @username with space
    text=re.sub('\ |\,|\`|\\|\/|\&|\*|\'|\;|\(|\)|\:|\/|\=|\-|\%|\$', ' ', text)    
    text = re.sub(r'#[a-zA-Z0-9]*', '', text)
    return text  

def Remove_Retweets(text):
	text=re.sub(r'\bRT\b', '', text)
	return text

# Strip space, " and ' from tweet
def Remove_Quotes(text):
	text = text.strip(' "\'')	
	# Remove - & 'text = re.sub(r'(-|\')', '', text)
	return text

def Remove_Emojis(text):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' ', text)
    return text

def Remove_Numbers(text):
    text = re.sub(r'[0-9]+', '', text)
    return text

def Remove_Capital_Words(text):
    text = re.sub(r'\b[A-Z]+\b', '', text)
    return text

def Remove_All_Dots(text):
    text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)
    text = Remove_Extra_Whitespace(text)
    return text

def Remove_extra(text):
    #text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)
    text = text.replace('\n', '').replace('\r', ' ').replace('\\', ' ').replace('@', ' ').replace('<', ' ').replace('>', ' ').replace('_', ' ')
    text = text.replace('[', '').replace(']', ' ').replace('~', ' ').replace("\"", "").replace("\'", "").replace("+", ' ').replace("-", ' ').replace("*", ' ')
    return text    

def Preprocessing_Start(raw_text):
    preprocessed_text=  Expand_Contractions(raw_text) 
    #preprocessed_text=  Eliminate_Capital_Words(preprocessed_text)
    preprocessed_text = p.clean(Strip_Links(preprocessed_text))        
    preprocessed_text = Remove_Ampersand(preprocessed_text)  # This function is called to remove ampersand (&amp) in raw text.    
    preprocessed_text = Remove_Hexchar(preprocessed_text)  # This function is  called to Remove hex char in raw text.    
    #preprocessed_text = Remove_Retweets(preprocessed_text)  # This function is  called to Remove hex char in raw text.
    preprocessed_text = ReplaceTwoOrMore(preprocessed_text)  # This function is  called to Remove hex char in raw text.  
    preprocessed_text = Remove_Retweets(preprocessed_text)  # This function is  called to Remove hex char in raw text.    
    preprocessed_text = Remove_charaters(preprocessed_text)  # This function is  called to Remove hex char in raw text.       
    preprocessed_text = Remove_Emojis(preprocessed_text) # This function is called to remove white spaces in raw text.
    preprocessed_text = Remove_Quotes(preprocessed_text) # This function is called to remove white spaces in raw text.    
    preprocessed_text = Remove_extra(preprocessed_text)    
    preprocessed_text = Remove_Numbers(preprocessed_text) # This function is called to remove white spaces in raw text.
    preprocessed_text = Convert_Lowercase(preprocessed_text)  # This function is  called to convert lower case letter in raw text.
    preprocessed_text = Remove_All_Dots(preprocessed_text) # This function is  called to remove all dots in raw text.   
    preprocessed_text = Remove_Extra_Whitespace(preprocessed_text) # This function is called to remove white spaces in raw text.    
    preprocessed_text = Stop_Words(preprocessed_text) # This function is called to remove stop words in raw text.    
    return preprocessed_text
    
#preprocessed_text=Preprocessing_Start(raw_tweet)
#print (preprocessed_text)





