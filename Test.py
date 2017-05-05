import os
import random
from pandas import json

from SentimentAnalysis import sentiment_analyzer_dict_complete
from SentimentAnalysis import sentiment_analyzer_dict
from SentimentAnalysis import sentiment_analyzer_classification


def read_data(folder):
    NB_Reviews_Class = 1000  # number of reviews of each class
    dataset = {  # learn about the composition of dataset
        '1': 0,  # positive
        '2': 0,  # positive
        '3': 0,  # neutral that will be ignored
        '4': 0,  # negative
        '5': 0   # negative
    }

    reviews = list()
    filenames = os.listdir(folder)
    for file in filenames:
        #if we have enough reviews, we stop the loop
        dataset_ready = [ dataset[rate_class] >=NB_Reviews_Class for rate_class in ['1','2','4','5']]
        if(dataset_ready.count(True) == 4):
            break

        # load the reviews
        file = folder + "/" + file
        try:
          with open(file, 'r') as f:
            data = json.load(f)
          for review in data['Reviews']:
            content = review['Content']
            rate = int(float(review['Overall']))
            if(rate==3):
                continue   # we ignore the neutral review whose rate is 3
            if (dataset[str(rate)] >= NB_Reviews_Class):  # make sure that the numbers of all class are sames
                continue
            else:
                dataset[str(rate)] += 1
            if(type(content)==str and type(rate)==int):
                reviews.append([content,rate])
        except Exception as e:
          print("Read file error: " + file)

    print("dataset of reviews" + str(dataset))
    return reviews

def divide_data(reviews):
    train_reviews = list()
    dev_reviews = list()
    test_reviews = list()
    random.shuffle(reviews)
    nb = len(reviews)
    trainPourcent = 0.75
    devPourcent = 0.15
    testPourcent = 0.15
    for i in range(0, nb):
        if (float(i / nb) <= trainPourcent):
            train_reviews.append(reviews[i])
        elif (float(i / nb) <= (trainPourcent + devPourcent)):
            dev_reviews.append(reviews[i])
        else:
            test_reviews.append(reviews[i])
    return train_reviews, dev_reviews, test_reviews



reviews= read_data("./cameras")
train_reviews, dev_reviews, test_reviews = divide_data(reviews)

# print("\nAnalyzer based on the one whole texte of review")
# analyzerWholeTexte = sentiment_analyzer_texte.sentiment_analyzer_texte()
# analyzerWholeTexte.train(train_reviews)
# analyzerWholeTexte.evaluate(dev_reviews)
# analyzerWholeTexte.get_score()

print("\nAnalyzer based on the sentences of review using dictionary of emotional words ")
analyzerSentenceDict = sentiment_analyzer_dict.sentiment_analyzer_dict()
analyzerSentenceDict.evaluate(dev_reviews)
analyzerSentenceDict.get_score()

print("\nAnalyzer based on the sentences of review using dictionary of emotional words with semantic analyse ")
analyzerSentenceDictComplete = sentiment_analyzer_dict_complete.sentiment_analyzer_dict_complete()
analyzerSentenceDictComplete.evaluate(dev_reviews)
analyzerSentenceDictComplete.get_score()


