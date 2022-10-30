from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	def tokenize(self, text):

	    #regular expression
		text = compile(r'<[^>]+>').sub(' ', text)
	    #to lowercase
		text = text.lower()
	    #remove punctuation and numbers
		text = sub('[^a-zA-Z]', ' ', text)
	    # remove single character 
		text = sub(r"\s+[a-zA-Z]\s+", ' ', text)
    	# remove multiple spaces
		text = sub(r'\s+', ' ', text)
    	# tokenization 
		tokens = text.split(' ')
		tokens = [i for i in tokens if i != '']
    	
		return tokens

	def build_vocab(self,corpus):
		print("def build_vocab(self, corpus) =========================")
		print()
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """ 

		appended_tokens = []
		for cor in corpus:
			appended_tokens += self.tokenize(cor)
		freq = Counter(appended_tokens)
		whole_tokens = list(freq.keys())

		freq_tokens = {key : value for key, value in freq.items() if value > 50 and key not in STOPWORDS}

		freq_words = list(freq_tokens.keys())

		word2idx = {}
		idx2word = {}

		for idx in range(len(freq_words)):
			word2idx[freq_words[idx]] = idx
			idx2word[idx] = freq_words[idx]

		word2idx['UNK'] = len(freq_words)
		idx2word[len(freq_words)] = 'UNK'

		return word2idx, idx2word, freq

	def make_vocab_charts(self):
		
		#sort with reverse order
		sorted_tokens = dict(sorted(self.freq.items(), key = lambda x:x[1], reverse = True))

		freq_sorted = list(sorted_tokens.values())

		fig1, ax1 = plt.subplots()
		ax1.plot(freq_sorted)
		ax1.hlines(y=50, xmin=0.0, xmax =60400, color = 'r', label = 'freq=50')
		ax1.text(55000, 75, 'freq=50', ha='center', va='center', color = 'r')
		ax1.set_yscale('log')
		ax1.set_ylim(bottom = 0, top = 10**6)
		ax1.set_title('Token Frequency Distribution')
		ax1.set_xlabel('Token ID (sorted by frequency)')
		ax1.set_ylabel('Frequency')

		#plt.show()
		plt.savefig("token_frequency_distribution.png", format='png')


		fig2, ax2 = plt.subplots()
		cumulative = np.cumsum(freq_sorted)
		freq_sorted_ratio = [freq / cumulative[-1] for freq in cumulative]
		ax2.plot(freq_sorted_ratio)
		ax2.set_yticks(np.arange(0.0, 1.1, 0.2))

		val = np.argmin(np.abs(np.array(freq_sorted_ratio)-0.92))

		ax2.vlines(x=val, ymin=0.0, ymax=1.0, color= 'r')
		ax2.text(9082, 0.9, 'ratio = 0.92', ha='left', va='center', color = 'r')
		ax2.set_title('Cumulative Fraction Covered')
		ax2.set_xlabel('Token ID (sorted by frequency)')
		ax2.set_ylabel('Fraction of Token Occurrences Covered')

		#plt.show()
		plt.savefig("cumulative_fraction_covered.png", format='png')
