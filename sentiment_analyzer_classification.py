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

        self.score = {
            'TP': 0,
            'TN': 0,
            'FN': 0,
            'FP': 0,
            'ENTIRE': 0
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
                self.score['TP'] += 1
            elif (pos >= neg and rate < 3):
                self.score['FP'] += 1
            elif (pos < neg and rate < 3):
                self.score['TN'] += 1
            elif (pos < neg and rate > 3):
                self.score['FN'] += 1
            else:
                print("Error")
        self.score['TP'] /= nb
        self.score['FP'] /= nb
        self.score['TN'] /= nb
        self.score['FN'] /= nb
        self.score['ENTIRE'] =  self.score['TP'] +  self.score['TN']

    def get_score(self):
        print("TP:%f"%self.score['TP'])
        print("FP:%f"%self.score['FP'])
        print("TN:%f"%self.score['TN'])
        print("FN:%f"%self.score['FN'])
        print("score entire:%f"%self.score['ENTIRE'])
        return self.score['ENTIRE']

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

        self.score = {
            'TP': 0,
            'TN': 0,
            'FN': 0,
            'FP': 0,
            'ENTIRE': 0
        }




