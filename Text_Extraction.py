# This script takes text from a .txt file and filters out/counts key words and phrases
# Original intention is to run this on research docs to quickly filter out major concepts covered in each file

# Import necessary libraries

import os
import nltk

# Punctuation and Stopwords package
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
stemmer = PorterStemmer()

# Variables

#STEP 1
# This is the variable name for the target file to read. Note it is useful to copy and paste all from 
# .PDF into a .TXT file to read
File_to_Read = 'Sample_from_PDF.txt'


# Read file
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

# Read file
corpus = PlaintextCorpusReader(os.getcwd(), File_to_Read)
#print(corpus.raw())

# Counts total sentences in document and creates a list of words in document
sentences = corpus.sents()
print("\n Total sentences in this corpus : ", len(sentences))
print("\n Words in this corpus : ", corpus.words())

# Finds frequency distribution of words in document
course_freq_dist = nltk.FreqDist(corpus.words())
print("\n Top 30 words in the corpus : ", course_freq_dist.most_common(30))

# Calculate distribution for a specific word
print("\n Distribution for \"hydrogen\" : ", course_freq_dist.get('hydrogen'))

# Tokenization

# Read base file into raw text variable
base_file = open(os.getcwd() + "/" + File_to_Read, mode = 'rt', encoding = 'utf-8')
raw_text = base_file.read()
base_file.close()

# Extract tokens
token_list = nltk.word_tokenize(raw_text)
print("Token List : ", token_list[30])
print("\n Total Tokens : ", len(token_list))

# Use punkt library to extract tokens while excluding punctuation
token_list2 = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, token_list))
print("Token list (top 50) after removing punctuation : ", token_list2[:50])
print("\n Total tokens after removing punctuation : ", len(token_list2))

# Convert to lowercase
token_list3 = [word.lower() for word in token_list2]
print("Token list after converting to lower case : ", token_list3[:50])
print("\nTotal tokens after converting to lower case :", len(token_list3))

# stopwords


# Remove stopwords
token_list4 = list(filter(lambda token: token not in stopwords.words('english'), token_list3))
print("Token list after removing stop words : ", token_list4[:50])
print("\n Total tokens after removing stopwords : ", len(token_list4))


# STEP 2: Modify list below if you want to exclude other common words that are showing up in your results. Typically this will include
# author names, dates, and various other. Not sure why some show up so many times but probably due to metadata of the .PDF file that
# got copied over into the .TXT file
# Create custom stopword list
custom_stopword_list = ['1062', 'environ', 'sci.', '2018', '11' , '1062', '1176', 'journal', 'royal' , 
                        'society', 'chemistry' , 'cite', 'way', 'mai', 'bui', 's.', 'ab', 'claire', 
                       's.', 'adjiman', 'andre' ,'bardow' , 'edward', 'j.', 'anthony', 'e' ,'andy'
                       'boston', 'solomon', 'brown', 'isâ©the','bc', 'andreâ´', 'andy', 'boston', 'f',
                        'g', 'paul', 'fennell', 'c', 'sabine', 'fuss', 'h', 'amparo', 'galindo', 'bc',
                        'leigh', 'a.', 'hackett', 'jason', 'p.', 'hallett', 'c', 'howard', 'herzog', 'j', 
                        'george', 'jackson', 'jasmin', 'kemper', 'k', 'samuel', 'krevor', 'lm', 'geoffrey', 
                        'c.', 'maitland', 'cl', 'michael', 'matuszewski', 'n', 'ian', 'metcalfe', 'camille',
                        'petit', 'graeme', 'puxty', 'p', 'jeffrey', 'reimer', 'q', 'david', 'm.', 'reiner', 
                        'r', 'rubin', 'stuart', 'scott', 'nilay', 'shah', 'berend', 'smit', 'qu', 'martin',
                        'trusler', 'cl', 'webley', 'vw', 'jennifer', 'wilcoxx', 'niall', 'mac', 'dowell']

token_list4a = list(filter(lambda token: token not in custom_stopword_list, token_list4))
print("Token list after removing additional CUSTOM stop words in custom list :", token_list4a[:80])
print("\n Total tokens after removing CUSTOM stopwords : ", len(token_list4a))


# Word frequency distribution AFTER removing custom words
course_freq_dist2 = nltk.FreqDist(token_list4a)
print("Top 80 words in the corpus : ", course_freq_dist2.most_common(80))

# Stem data - reduce words to root form
token_list5 = [stemmer.stem(word) for word in token_list4a ]
print("Token list after stemming :", token_list5[:80] )
print("\n Total tokens after Stemming : ", len(token_list5))

#Lemmatizing - groups together inflected forms of a word (i.e. confidence and confidently)

lemmatizer = WordNetLemmatizer()
token_list6 = [lemmatizer.lemmatize(word) for word in token_list4a]
print("Token list after Lemmaization : ", token_list6[:80])
print("\n Total tokens after Lemmatization : ", len(token_list6))

#n-grams - find the most common 2- and 3-word phrases


#Find bigrams and print most common 40
bigrams = ngrams(token_list6,2)
print("Most common bigrams : ")
print(Counter(bigrams).most_common(80))


#Find tri-grams and print most common 40
trigrams = ngrams(token_list6,3)
print("Most common trigrams : ")
print(Counter(trigrams).most_common(80))
