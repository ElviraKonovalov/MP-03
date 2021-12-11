import gensim.downloader as api
import os
import pandas as pd
import random

#=======================================================#
#                      functions                        #
#=======================================================#

def get_model_stats(model_name, model_words, question_words, answer_words, guess_words):
	guess_num = 0
	correct_num = 0
	wrong_num = 0

	details_file = model_name + "-details.csv"

	with open(details_file, 'a') as f:
		f.writelines("question-word,correct answer-word,systems guess-word,label\n")
		for index, word in enumerate(question_words):
			system_guess_word = ""
			similairy = 0
			# print("all possible guesses: " + str(guess_words[index, :]))
			if word not in model_words or guess_words[index, :].all() not in model_words or (word and guess_words[index, :].all()) not in model_words:
				# print("{},{},-,guess".format(word, answer_words[index]))
				f.writelines("{},{},-,guess\n".format(word, answer_words[index]))
				guess_num += 1
			else:
				for guess in guess_words[index, :]:
					try:
						# print("'{}' '{}' {}".format(word, guess, round(model.similarity(word, guess), 4)))
						if similairy < model.similarity(word, guess):
							similairy = model.similarity(word, guess)
							system_guess_word = guess
					except KeyError:
						print("'{}' is not in model_words!".format(guess))
				if system_guess_word == answer_words[index]:
					# print("{},{},{},correct".format(word, answer_words[index], system_guess_word))
					f.writelines("{},{},{},correct\n".format(word, answer_words[index], system_guess_word))
					correct_num += 1
				else:
					# print("{},{},{},wrong".format(word, answer_words[index], system_guess_word))
					f.writelines("{},{},{},wrong\n".format(word, answer_words[index], system_guess_word))
					wrong_num += 1

	accuracy = round((correct_num/(correct_num+wrong_num)),5)

	analysis = "{},{},{},{},{}\n".format(model_name,len(set(model_words)),correct_num,(80-guess_num),accuracy)
	print("{},{},{},{},{}".format(model_name,len(set(model_words)),correct_num,(80-guess_num),accuracy))

	with open('analysis.csv', 'a') as f:
		f.writelines(analysis)

#=======================================================#
#                      Task 0                           #
#=======================================================#
dirname = os.path.dirname(__file__)
filename = 'synonyms.csv'
file = os.path.join(dirname, filename)

df = pd.read_csv(file)

# extract questions, correct answers and 
question_words = df.iloc[:,0].values
answer_words = df.iloc[:, 1].values
guess_words = df.iloc[:, 2:].values

with open('analysis.csv', 'a') as f:
	f.writelines("model name,vocabulary size,C,V,accuracy\n")

#=======================================================#
#                      Task 1                           #
#=======================================================#

model_name = "word2vec-google-news-300"
model = api.load(model_name) 

model_words = []	
for index, word in enumerate(model.index_to_key):
	model_words.append(word)

get_model_stats(model_name, model_words, question_words, answer_words, guess_words)

#=======================================================#
#                      Task 2                           #
#=======================================================#

#---------------------------------------------------------------------#
# (a) 2 new models from different corpora but same embedding size     #
#---------------------------------------------------------------------#

# first model with embedding size 300
model_name = "fasttext-wiki-news-subwords-300"
model = api.load(model_name) 

model_words = []	
for index, word in enumerate(model.index_to_key):
	model_words.append(word)

get_model_stats(model_name, model_words, question_words, answer_words, guess_words)

# second model with embedding size 300
model_name = "glove-wiki-gigaword-300"
model = api.load(model_name) 

model_words = []	
for index, word in enumerate(model.index_to_key):
	model_words.append(word)

get_model_stats(model_name, model_words, question_words, answer_words, guess_words)

#---------------------------------------------------------------------#
# (b) 2 new models from the same corpus but different embedding sizes #
#---------------------------------------------------------------------#

# model with with embedding size 50
model_name = "glove-twitter-50"
model = api.load(model_name) 

model_words = []	
for index, word in enumerate(model.index_to_key):
	model_words.append(word)

get_model_stats(model_name, model_words, question_words, answer_words, guess_words)

# same model with embedding size 200
model_name = "glove-twitter-200"
model = api.load(model_name) 

model_words = []	
for index, word in enumerate(model.index_to_key):
	model_words.append(word)

get_model_stats(model_name, model_words, question_words, answer_words, guess_words)


#---------------------------------------------------------------------#
# (c) random baseline 												  #
#---------------------------------------------------------------------#
random.seed(0)

guess_num = 0
correct_num = 0
wrong_num = 0

model_name = "random_baseline"

details_file = model_name + "-details.csv"

with open(details_file, 'a') as f:
	f.writelines("question-word,correct answer-word,systems guess-word,label\n")
	for index, word in enumerate(question_words):
		random_num = random.randint(0,3)
		random_guess = guess_words[index, random_num]
		system_guess_word = random_guess

		if system_guess_word == answer_words[index]:
			f.writelines("{},{},{},correct\n".format(word, answer_words[index], system_guess_word))
			correct_num += 1
		else:
			f.writelines("{},{},{},wrong\n".format(word, answer_words[index], system_guess_word))
			wrong_num += 1

accuracy = round((correct_num/(correct_num+wrong_num)),5)

analysis = "{},{},{},{}\n".format(model_name,correct_num,(80-guess_num),accuracy)
print(analysis)
