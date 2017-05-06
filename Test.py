import os
import random

import math
from pandas import json

import sentiment_analyzer_dict
import sentiment_analyzer_dict_complete
import sentiment_analyzer_classification


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
          print("Ignored Read file error: " + file)

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


print(">>>>>Preparation TEST")
reviews= read_data("./cameras")
train_reviews, dev_reviews, test_reviews = divide_data(reviews)

# exit(0)

analyzerClassification = sentiment_analyzer_classification.sentiment_analyzer_classfication()
analyzerClassification.train(train_reviews)

analyzerSentenceDictComplete = sentiment_analyzer_dict_complete.sentiment_analyzer_dict_complete()

print("===============Performance TEST==============")
print("\n>>Analyzer based on the classification algo Bayes")
analyzerClassification.evaluate(dev_reviews)
analyzerClassification.get_score()

print("\n>>Analyzer based on the dictionary of emotional words with semantic analyse")
analyzerSentenceDictComplete.evaluate(dev_reviews)
analyzerSentenceDictComplete.get_score()

print("===============Precise TEST==============")
sentences = {
    "Good and not good",  # neutral sentence, used in the all sentence below for the reason of a good visual comparison
    "This camera have a good price and easy to use. Good and not good",  # positive sentence
    "This camera is bad and difficult to use.  Good and not good",         #  negative sentence
    "This camera don't have a good price and easy to use. Good and not good",  # negative sentence
    "This camera have a good price and easy to use! Good and not good",  # punctuation emphize
    "This camera have a very good price and easy to use. Good and not good",  # degree word
    "This camera don't have a good price but easy to use! Good and not good",  # conjuction "but"
}
print("\n>>Analyzer based on the classification algo Bayes")
for sentence in sentences:
    pos, neg = analyzerClassification.predict_review(sentence)
    # transform the log prob
    normal =  pos + abs(min(pos,neg)) # the two log pro < 0 , and very small, we add the bias
    pos_normal = math.pow(math.e, pos + normal)
    neg_normal = math.pow(math.e, neg + normal)
    # normoalize two value, pos + neg = 1
    unit = pos_normal + neg_normal
    pos_normal = pos_normal/unit
    neg_normal = neg_normal/unit
    print(sentence + "\npositive:" + str(pos_normal) + "," + "negative:" + str(neg_normal))

print("\n>>Analyzer based on the dictionary of emotional words with semantic analyse")
for sentence in sentences:
    pos, neg = analyzerSentenceDictComplete.predict_review(sentence)
    print(sentence + "\npositive:" + str(pos) + "," + "negative:" + str(neg))

