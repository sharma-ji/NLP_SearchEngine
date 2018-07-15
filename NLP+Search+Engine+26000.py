
# coding: utf-8

# # Creating a Simple Search Engine using NLP

# **We will start by importing the required libraries**

# In[2]:

import codecs 
import re
import numpy as np
from IPython.display import clear_output


# In[3]:

## Let us load the dataset 


# In[4]:

text = codecs.open('./wiki-26000', encoding='utf-8').read()


# In[ ]:

starts = [match.span()[0] for match in re.finditer('\n = [^=]', text)]


# In[ ]:

articles = list()


# In[ ]:

for ii, start in enumerate(starts):
    end = starts[ii+1] if ii+1 < len(starts) else len(text)
    articles.append(text[start:end])


# In[ ]:

snippets = [' '.join(article[:300].split()) for article in articles]
 
for snippet in snippets[:20]:
    print(snippet)


# Some notes:
# 
# 1. We are using the wiki-600 to begin with.
# 2. All articles are in one file. Articles titles are formatted as follows: = Albert Einstein = . If it has two or more =      signs on both sides, then it's a subheading. The regex looks for article titles, and splits the text file.
# 3. Then we calculate snippets, i.e. the first 300 characters for each article.

# In[ ]:

# Tokenising the article 
# Calculating Term Frequencies
import sys
from collections import defaultdict
from nltk.tokenize import word_tokenize               # <=== tokenizer 
from nltk.stem.porter import PorterStemmer            # <=== stemmer 
from nltk.corpus import stopwords as nltk_stopwords   # <=== stopwords
STOPWORDS = set(nltk_stopwords.words('english'))


# 1. We start by initiating PorterStemmer, it is used to normalize words for e.g. Loves, Love, loving, loved all these words give the same context, but their multiple occurance gives more vectors and more computation is required. So its better to leave them
# 2. Inside the function, we first lower case the words to have all the words in same patter
# 3. Then we remove everything excecpt Alphanumerics
# 4. We split and join to give a string again
# 5. Then we remove the stopwords
# 6. In the end we have cleaned dataset with only aplhanumerics

# In[ ]:

ps = PorterStemmer()
term_frequency = defaultdict(dict)
 
def get_tokens(article):
    article = article.lower()
    article = re.sub(r'[^a-zA-Z0-9]', ' ', article)
    article.split()
    article = ''.join(article)
    tokens = [ps.stem(tokens) for tokens in article.split() if tokens not in nltk_stopwords.words("english")]
    return tokens


# In the index function, we first tokenize the given article, then we count the number of occurance of every token and store it in the term frequency dictionary
# 
# Then using for loop we populate the term_frequency dictionary and it gives us term frequency of corpus

# In[ ]:

def index(id, article):
    tokens = get_tokens(article)
    # TODO: calculate term frequencies and store in term_frequency[token][id]
    for token in tokens:
        term_frequency[token][id] = tokens.count(token)
 
for ii, article in enumerate(articles):
    if ii and ii % 10 == 0: print(ii, end=', ')
    sys.stdout.flush()
    index(ii, article)


# Just a random check of our function

# In[ ]:

print('term_frequency for "einstein"')
print(term_frequency['einstein'])
print(len(term_frequency['einstein']))
# Expected output: {300: 1, 84: 5, 294: 1}
# That is, articles[300] has token einstein 1 times, articles[5] has 
# token einstein 5 times, and articles[294] has token einstein 1 times.


# This function is used to count **TFIDF**, here we are calling the **term_frequncy** for the particular word. And then counting the articles which contain the word(query). And using these values we are calculating the TFIDF 

# In[ ]:

def TFIDF(article_id, query):
    tf = term_frequency[query][article_id]
    total_articles = len(snippets)
    desired_articles = len(term_frequency[query])
    if desired_articles!=0:
        tfidf = tf*np.log(total_articles/desired_articles)
    else:
        tfidf = tf*np.log(total_articles/1)
    return tfidf
    


# ### Saving the data in pickle file

# In[ ]:

###########################################################################
## saving and loading
 
import pickle
 
def picklesave(obj, filename):
    print('Saving .. ')
    ff = open(filename, 'wb')
    pickle.dump(obj, ff)
    ff.close()
    print('Done')
    return True
 
def pickleload(filename):
    print('Loading .. ')
    ff = open(filename, 'rb')
    obj = pickle.load(ff)
    ff.close()
    print('Done')
    return obj
 


# In[ ]:

picklesave([snippets, term_frequency], 'data-26000.pdata')
snippets, term_frequency = pickleload('data-26000.pdata')


# ## Ranking the articles for Search!
# It's time to write the final search function.

# In[ ]:

import math

def sort(dictionary):
    m= defaultdict(float)
    for w in sorted(dictionary, key=dictionary.get, reverse=True):
        m[w]=dictionary[w]
    return m
D = len(snippets)
def search(query, nresults=10):
    tokens = get_tokens(query)
    scores = defaultdict(float)
    for token in tokens:
        for article, score in term_frequency[token].items():
            scores[article] = TFIDF(article,token) 
    new_scores = sort(scores)
    return new_scores# TODO: top nresults results
 
def display_results(query, results):
    print('You search for: "%s"' % query)
    print('-'*100)
    for result in results:
        print(snippets[result])
    print('='*100)


# **The interactive search bar**

# In[ ]:

###########################################################################
## interactive
while True:
    clear_output()
    query = input("Please enter the search query: \nEnter q to quit")
    display_results(query, search(query))
    if query=='q':
        break


# In[ ]:



