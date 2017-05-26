import math

import nltk


class sentiment_analyzer_classfication:
    SMOOTHING = 1

    def __init__(self):
        self.initialization()

    def initialization(self):
        self.proWords = {
            "negative": dict(),  # P(w|c='negative')
            "positive": dict()  # P(w|c='positive')
        }

        self.proClass = {
            "negative": 0.0,  # P(C='negative')
            "positive": 0.0  # P(C='positive')
        }

        self.count_reviews = {
            "negative": 0,  # total number of all positive reviews
            "positive": 0  # total number of all negative reviews
        }

        self.count_words = {
            "negative": 0,  # total number of all positive words
            "positive": 0  # total number of all negative words
        }

        self.accuracy = {
            'TP': 0,
            'TN': 0,
            'FN': 0,
            'FP': 0,
            'ACCURACY': 0
        }

    def train(self, train_reviews):
        self.initialization()
        for review in train_reviews:
            content = review[0]
            rate = review[1]
            if (rate > 3):  # nagative sentence: rate = 4 or 5
              self.count_reviews['positive'] += 1
            elif (rate < 3):  # positive sentence: rate = 1 or 2
              self.count_reviews['negative'] += 1

            allwords = nltk.word_tokenize(content)
            for word in set(allwords):  # Word occurrence may matter more than word frequency
              if (rate > 3):  # nagative review: rate = 4 or 5
                  if (word in self.proWords["positive"].keys()):
                      self.proWords["positive"][word] += 1
                  else:
                      self.proWords["positive"][word] = 1
              elif (rate < 3):  # positive review: rate = 1 or 2
                  if (word in self.proWords["negative"].keys()):
                      self.proWords["negative"][word] += 1
                  else:
                      self.proWords["negative"][word] = 1

        # calcul the log probability of each class
        self.calculate_class_logprobas()

        # transform the frequency of the word into its log porbability P(w|y=c)
        self.calculate_word_logprobas()


    def calculate_class_logprobas(self):
      nb_sentences = self.count_reviews['positive']+self.count_reviews['negative']
      self.proClass['positive'] = math.log((self.count_reviews['positive']+self.SMOOTHING)/(nb_sentences+self.SMOOTHING))
      self.proClass['negative'] = math.log((self.count_reviews['negative']+self.SMOOTHING)/(nb_sentences+self.SMOOTHING))

    def calculate_word_logprobas(self):
      for count_word in self.proWords['positive'].values():
        self.count_words['positive'] += count_word
      for count_word in self.proWords['negative'].values():
        self.count_words['negative'] += count_word

      for word in self.proWords['positive'].keys():
        self.proWords['positive'][word]= math.log(self.proWords['positive'][word] / self.count_words['positive'])
      for word in self.proWords['negative'].keys():
        self.proWords['negative'][word] = math.log(self.proWords['negative'][word] / self.count_words['negative'])

    def get_logproba_review(self,content):
      pro_positive = 0
      pro_negative = 0
      count_UNK_positive = 0
      count_UNK_negative = 0
      words = nltk.word_tokenize(content)
      words = set(words) # Word occurrence may matter more than word frequency
      #count the number of words who never appear in the train data set
      for word in words:
          if word not in self.proWords['positive'].keys():
              count_UNK_positive += 1
          if word not in self.proWords['negative'].keys():
              count_UNK_negative += 1

      for word in words:
          if word in self.proWords['positive'].keys():
              pro_positive += self.proWords['positive'][word]
          else:
              pro_positive += math.log((0 + self.SMOOTHING) / (self.count_words['positive'] + count_UNK_positive * self.SMOOTHING))

          if word in self.proWords['negative'].keys():
              pro_negative += self.proWords['negative'][word]
          else:
              pro_negative += math.log((0 + self.SMOOTHING) / (self.count_words['negative'] + count_UNK_negative * self.SMOOTHING))

      pro_positive += self.proClass['positive']
      pro_negative += self.proClass['negative']
      return pro_positive,pro_negative


    def predict_review(self,content):
        pro_pos, pro_neg = self.get_logproba_review(content)
        return pro_pos,pro_neg

    def evaluate(self,dev_reviews):
        nb = len(dev_reviews)
        for review in dev_reviews:
            pos, neg = self.predict_review(review[0])
            rate = review[1]
            if (pos >= neg and rate > 3):
                self.accuracy['TP'] += 1
            elif (pos >= neg and rate < 3):
                self.accuracy['FP'] += 1
            elif (pos < neg and rate < 3):
                self.accuracy['TN'] += 1
            elif (pos < neg and rate > 3):
                self.accuracy['FN'] += 1
            else:
                print("Error")
        # self.accuracy['TP'] /= nb
        # self.accuracy['FP'] /= nb
        # self.accuracy['TN'] /= nb
        # self.accuracy['FN'] /= nb
        self.accuracy['ACCURACY'] = (self.accuracy['TP'] + self.accuracy['TN'])/(self.accuracy['TP'] + self.accuracy['TN']+self.accuracy['FP']+self.accuracy['FN'])

    def get_accuracy(self):
        print("TP:%d" % self.accuracy['TP'])
        print("FP:%d" % self.accuracy['FP'])
        print("TN:%d" % self.accuracy['TN'])
        print("FN:%d" % self.accuracy['FN'])
        print("Accuracy:%f" % self.accuracy['ACCURACY'])
        return self.accuracy['ACCURACY']

    def reset(self):
        self.proWords = {
            "negative": dict(),  # P(w|c='negative')
            "positive": dict()  # P(w|c='positive')
        }

        self.proClass = {
            "negative": 0.0,  # P(C='negative')
            "positive": 0.0  # P(C='positive')
        }

        self.count_reviews = {
            "negative": 0,  # total number of all positive reviews
            "positive": 0  # total number of all negative reviews
        }

        self.count_words = {
            "negative": 0,  # total number of all positive words
            "positive": 0  # total number of all negative words
        }

        self.accuracy = {
            'TP': 0,
            'TN': 0,
            'FN': 0,
            'FP': 0,
            'ACCURACY': 0
        }


def main():
    import os
    import random

    import math
    import json

    def read_data(folder):
        NB_Reviews_Class = 1000  # number of reviews of each class
        dataset = {  # learn about the composition of dataset
            '1': 0,  # positive
            '2': 0,  # positive
            '3': 0,  # neutral that will be ignored
            '4': 0,  # negative
            '5': 0  # negative
        }

        reviews = list()
        filenames = os.listdir(folder)
        for file in filenames:
            # if we have enough reviews, we stop the loop
            dataset_ready = [dataset[rate_class] >= NB_Reviews_Class for rate_class in ['1', '2', '4', '5']]
            if (dataset_ready.count(True) == 4):
                break
            # load the reviews
            file = folder + "/" + file
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                for review in data['Reviews']:
                    content = review['Content']
                    rate = int(float(review['Overall']))
                    if (rate == 3):
                        continue  # we ignore the neutral review whose rate is 3
                    if (dataset[str(rate)] >= NB_Reviews_Class):  # make sure that the numbers of all class are sames
                        continue
                    else:
                        dataset[str(rate)] += 1
                    if (type(content) == str and type(rate) == int):
                        reviews.append([content, rate])
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

    reviews = read_data("./cameras")
    train_reviews, dev_reviews, test_reviews = divide_data(reviews)

    analyzerClassification = sentiment_analyzer_classfication()
    analyzerClassification.train(train_reviews)
    print("\n>>Analyzer based on the classification algo Bayes")

    import sys
    for sentence in sys.argv[1:]:
        pos, neg = analyzerClassification.predict_review(sentence)
        # transform the log prob
        normal = pos + abs(min(pos, neg))  # the two log pro < 0 , and very small, we add the bias
        pos_normal = math.pow(math.e, pos + normal)
        neg_normal = math.pow(math.e, neg + normal)
        # normoalize two value, pos + neg = 1
        unit = pos_normal + neg_normal
        pos_normal = pos_normal / unit
        neg_normal = neg_normal / unit
        print(sentence + "\npositive:" + str(pos_normal) + "," + "negative:" + str(neg_normal)+"\n")


if __name__ == "__main__":
    main()


