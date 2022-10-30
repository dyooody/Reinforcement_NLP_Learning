from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE

import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


def compute_cooccurrence_matrix(corpus, vocab):
	print("def compute_cooccurrence_matrix(corpus, vocab) =================")
	print()

	vocab_size = len(vocab.word2idx)
	window_size = 4

	print("length of corpus: ", len(corpus))

	print("size of matrix : ", vocab)

	co_matrix = np.zeros((vocab_size, vocab_size), np.int32)

	for text in corpus:
		text_idx = vocab.text2idx(text)

		for idx in range(len(text_idx)):
			cur_idx = text_idx[idx]

			mat_range = min(len(text_idx) - 1, idx + window_size) # for right search
			#mat_range = max(idx - window_size, 0) # for left search 

			#for jdx in ragne(mat_range, idx): # for left search
			for jdx in range(idx + 1, mat_range + 1):
				tar_idx = text_idx[jdx]

				if tar_idx != cur_idx:
					co_matrix[cur_idx][tar_idx] += 1
					co_matrix[tar_idx][cur_idx] += 1


	return co_matrix



def compute_ppmi_matrix(corpus, vocab):
	print("def compute_ppmi_matrix(corpus, vocab): =========================")
	print()

	co_mat = compute_cooccurrence_matrix(corpus, vocab)

	positive = True 

	col_totals = co_mat.sum(axis = 0)
	total = col_totals.sum()
	row_totals = co_mat.sum(axis = 1)
	expected = np.outer(row_totals, col_totals) / total

	ppmi_mat = co_mat / expected

	with np.errstate(divide = 'ignore'):
		ppmi_mat = np.log(ppmi_mat)

	ppmi_mat[np.isinf(ppmi_mat)] = 0.0 #log(0) = 0

	if positive:
		ppmi_mat[ppmi_mat < 0] = 0.0

	return ppmi_mat


def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")

	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]

	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)

	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)

	plt.show()
	#plt.savefig("plot_word_vectors_tsne.png", format='png')


if __name__ == "__main__":
    main_freq()

