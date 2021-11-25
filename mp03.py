import gensim.downloader as api
import os
import pandas as pd

model = api.load("word2vec-google-news-300")  # load glove vectors

model_words = []	
for index, word in enumerate(model.index_to_key):
	model_words.append(word)

dirname = os.path.dirname(__file__)
filename = 'synonyms.csv'
file = os.path.join(dirname, filename)

df = pd.read_csv(file)

# extract questions, correct answers and 
question_words = df.iloc[:,0].values
answer_words = df.iloc[:, 1].values
guess_words = df.iloc[:, 2:].values

guess_num = 0
correct_num = 0
wrong_num = 0

# for index, word in enumerate(question_words):
# 	try:
# 		system_guess_word = model.most_similar(word, topn=1)[0][0]
# 		if word not in model_words or guess_words[index, 2:].all() not in model_words or (word and guess_words[index, 2:].all()) not in model_words:
# 			print("{},{},-,guess".format(system_guess_word, answer_words[index]))
# 			guess_num += 1
# 		elif system_guess_word == answer_words[index]:
# 			print("{},{},{},correct".format(word, answer_words[index], system_guess_word))
# 			correct_num += 1
# 		else:
# 			print("{},{},{},wrong".format(word, answer_words[index], system_guess_word))
# 			wrong_num += 1
# 	except KeyError:
# 		print("{},{},-,guess".format(word, answer_words[index]))
# 		guess_num += 1

# accuracy = round((correct_num/(correct_num+wrong_num)),2)
# print("{},{},{},{},{}".format("word2vec-google-news-300",len(set(model_words)),correct_num,guess_num,accuracy))

for index, word in enumerate(question_words):
	system_guess_word = ""
	similairy = 0
	if word not in model_words or guess_words[index, :].all() not in model_words or (word and guess_words[index, :].all()) not in model_words:
		print("{},{},-,guess".format(system_guess_word, answer_words[index]))
		guess_num += 1
		break
	print("all possible guesses: " + str(guess_words[index, :]))
	for guess in guess_words[index, :]:
		try:
			print("'{}' '{}' {}".format(word, guess, round(model.similarity(word, guess), 4)))
			if similairy < model.similarity(word, guess):
				similairy = model.similarity(word, guess)
				system_guess_word = guess
		except KeyError:
			print("'{}' is not in model_words!".format(guess))
	if system_guess_word == answer_words[index]:
		print("{},{},{},correct".format(word, answer_words[index], system_guess_word))
		correct_num += 1
	else:
		print("{},{},{},wrong".format(word, answer_words[index], system_guess_word))
		wrong_num += 1

accuracy = round((correct_num/(correct_num+wrong_num)),2)
print("{},{},{},{},{}".format("word2vec-google-news-300",len(set(model_words)),correct_num,guess_num,accuracy))