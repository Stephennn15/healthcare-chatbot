# MeetPybot: your friend

import nltk
import warnings
warnings.filterwarnings("ignore")
# nltk.download() # for downloading packages
#import tensorflow as tf
import numpy as np
import random
import string  # to process standard python strings

# Loading data files
f = open('symptom.txt', 'r', errors='ignore')
m = open('pincodes.txt', 'r', errors='ignore')
checkpoint = "./chatbot_weights.ckpt"
# session = tf.InteractiveSession()
# session.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(session, checkpoint)

raw = f.read()
rawone = m.read()

# Downloading necessary NLTK packages
nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words
sent_tokensone = nltk.sent_tokenize(rawone)  # converts to list of sentences
word_tokensone = nltk.word_tokenize(rawone)  # converts to list of words

sent_tokens[:2]
sent_tokensone[:2]
word_tokens[:5]
word_tokensone[:5]

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Introduce_Ans = [" "]
GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii", "hiiii", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = [
    "Hi, are you suffering from any health issues?(Y/N)",
    "Hey, are you having any health issues?(Y/N)",
    "Hii there, are you having any health issues?(Y/N)"
]

Basic_Q = ("yes", "y", "yep", "yup")
Basic_Ans = "Describe your symptoms"
Basic_Om = ("no", "n", "nope")
Basic_AnsM = "Thanks for stopping by, we hope to see you again in the near future"
fev = ("iam suffering from fever", "i affected with fever", "i have fever", "fever")
feve_r = "Which type of fever do you have and please mention your symptoms"

# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Checking for Basic_Q
def basic(sentence):
    for word in Basic_Q:
        if sentence.lower() == word:
            return Basic_Ans

def fever(sentence):
    for word in fev:
        if sentence.lower() == word:
            return feve_r

# Checking for Basic_QM
def basicM(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Om:
        if sentence.lower() == word:
            return Basic_AnsM

# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = robo_response + "I'm sorry, I didn't quite understand what you said. Could you please try rephrasing it?"
    else:
        robo_response = robo_response + sent_tokens[idx]
    return robo_response

# Generating response for second set of tokens
def responseone(user_response):
    robo_response = ''
    sent_tokensone.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokensone)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
    else:
        robo_response = robo_response + sent_tokensone[idx]
    return robo_response

def chat(user_response):
    user_response = user_response.lower()
    keyword = " module "
    keywordone = " module"
    keywordsecond = "module"
    
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            return "You are welcome."
        elif basicM(user_response) != None:
            return basicM(user_response)
        else:
            if user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(keywordsecond) != -1:
                return responseone(user_response)
            elif greeting(user_response) != None:
                return greeting(user_response)
            elif user_response.find("yourname") != -1 or user_response.find("your name") != -1:
                return IntroduceMe(user_response)
            elif basic(user_response) != None:
                return basic(user_response)
            elif fever(user_response) != None:
                return fever(user_response)
            else:
                return response(user_response)
        
        sent_tokens.remove(user_response)
    else:
        flag = False
        return "Bye! Take care.."
