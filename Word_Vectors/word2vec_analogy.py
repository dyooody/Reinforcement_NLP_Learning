import gensim.downloader

w2v = gensim.downloader.load('word2vec-google-news-300')

def analogy(a, b, c):
	print(a+ " : "+b+" :: "+c+" : ?")
	print([(w, round(c, 3)) for w,c in w2v.most_similar(positive=[c,b], negative=[a])])
	print()


# Task 4.1 
print("===== Task 4.1 =====")
analogy('man', 'king', 'woman')
analogy('usa', 'dollar', 'EU')
analogy('Tokyo', 'Japan', 'Oslo')
analogy('brother', 'boy', 'sister')
analogy('Seoul', 'Korea', 'Pyeongyang')

# Task 4.2
print("===== Task 4.2 =====")
analogy('white', 'bright', 'black')
analogy('water', 'liquid', 'ice')
analogy('violin', 'string', 'trumpet')
analogy('chipmunk', 'rodent', 'giraffe')

# Task 4.3
print("===== Task 4.3 =====")
analogy('man', 'computer_programmer', 'woman')
analogy('woman', 'computer_programmer', 'man')
analogy('man', 'pilots', 'woman')
analogy('woman', 'pilots', 'man')