import re
import nltk

class sentiment_analyzer_dict_complete:
    NEGATIONS = list()
    ADVERBS_DEGREE = dict()

    def __init__(self):
        # as the dictionary of positive and negative words with their value
        self.dictionary = dict()
        self.score = {
            'TP': 0,
            'TN': 0,
            'FN': 0,
            'FP': 0,
            'ENTIRE': 0
        }

        with open('./SentiWordNet_simple_format.txt', 'r') as f:
            words_values = f.read().split('\n')
            for word_value in words_values:
                word_value=word_value.split(':')
                if(len(word_value)==2):
                    self.dictionary[word_value[0]]=float(word_value[1])

        with open('./negation.txt', 'r') as f:
            self.NEGATIONS=f.read().split("\n")

        with open('./degree.txt', 'r') as f:
            lines = f.read().split("\n")
            for adverb_degree in lines:
                adverb_degree = adverb_degree.split(":")
                self.ADVERBS_DEGREE[adverb_degree[0]] = adverb_degree[1]


    def mark_negation(self, words):
        mark = False
        for i, word in enumerate(words):
            if (word in self.NEGATIONS):
                mark = False if mark else True  # double negation.txt make affirmation
                words[i] = '' # we don't need the negation anymore
                continue
            if (mark):
                words[i] += '_NEG'
        # print(words)
        return words

    def sent_tokenize(self,content):
        # use nltk to get the sentences of one texte
        sents = nltk.sent_tokenize(content)
        for sentence in sents:
            # we do a very simple supplemental sepation to fix some simple defaut of nltk.sent_tokenize
            # for example,nltk.sent_tokenize don't seperate  [I like you.You like me.] ,where it think that there
            # should be a space between two sentences. It believe that [you.You] is one token.
            # Actually, the problem is very common in many clients reviews, because for many people,
            # they don't use spaces between the sentences.
            supplemental_separated = re.sub("([\.\!\?]+)", r"\1\n", sentence)
            supplemental_sents = supplemental_separated.strip().split("\n")
            sents.remove(sentence)
            sents.extend(supplemental_sents)
        return sents


    def predict_review(self,review_content):
      value_pos = 0
      value_neg = 0
      sentences = self.sent_tokenize(review_content)
      for sentence in sentences:  # each sentence is an unit , not the total texte
          tempPos, tempNeg = self.get_value_sentence(sentence)
          value_pos += tempPos
          value_neg += tempNeg

      # print("review",value_pos,value_neg)
      # normoalize two value, pos + neg = 1
      if (value_pos + value_neg != 0):
          pos = value_pos / (value_pos - value_neg)
          neg = -value_neg / (value_pos - value_neg)
          return pos,neg
      else:
          return 0.5, 0.5

    def get_value_sentence(self,sentence):
        value_pos = 0
        value_neg = 0
        words = nltk.word_tokenize(sentence)

        # turning logic
        if(words.count("but")==1):
            value_pos, value_neg = self.conjunction_but(sentence)
        else:
            value_pos, value_neg = self.get_value_words(words)

        # degree effected by punctuation of exclamation
        punction_emphize = self.punctuation_emphize(sentence)
        if(value_pos + value_neg >=0):
            value_pos *= punction_emphize
        elif(value_neg + value_neg <0):
            value_neg *= punction_emphize

        return value_pos, value_neg


    def get_value_words(self,words):
        value_pos = 0
        value_neg = 0
        words = self.mark_negation(words)
        degree_adverbe = 1
        for word in words:
          if(word.lower() in self.ADVERBS_DEGREE.keys()):  #for the degree adverb
              if(float(self.ADVERBS_DEGREE.get(word.lower()))>0):
                degree_adverbe = 1+float(self.ADVERBS_DEGREE.get(word.lower()))/1.5
              else:
                degree_adverbe = -1+float(self.ADVERBS_DEGREE.get(word.lower()))/1.5
              continue
          # print(degree_adverbe)

          negation = 1  # if the word is tagged by negation, we inverse its value
          if (re.search(r'(\w+)_NEG$', word) != None):
              negation = -1
              word = re.sub(r'(\w+)_NEG$', r'\1', word)
          if word.lower() in self.dictionary.keys():
              temp = self.dictionary.get(word.lower()) * negation * degree_adverbe
              if(temp>=0):
                value_pos += temp
              else:
                value_neg += temp
        return value_pos,value_neg


    def conjunction_but(self,sentence):
        value_pos = 0
        value_neg = 0
        words = nltk.word_tokenize(sentence)
        index = words.index("but")
        words_before = words[:index]
        words_after = words[index:]
        tempPos, tempNeg = self.get_value_words(words_before)
        tempPos2, tempNeg2 = self.get_value_words(words_after)
        value_pos += tempPos + 1.5 * tempPos2
        value_neg += tempNeg + 1.5 * tempNeg2
        return value_pos, value_neg


    def punctuation_emphize(self, sentence):
        words = nltk.word_tokenize(sentence)
        nb_excalmatory = words.count('!')
        if nb_excalmatory <= 3:
            return 1 + nb_excalmatory * 0.5
        else:
            return 3


    def evaluate(self,dev_reviews):
        nb = len(dev_reviews)
        for review in dev_reviews:
            pos,neg = self.predict_review(review[0])
            rate = review[1]
            if ( pos >= neg and rate > 3):
                self.score['TP'] += 1
            elif (pos >= neg and rate < 3):
                self.score['FP'] += 1
            elif (pos < neg and rate < 3):
                self.score['TN'] += 1
            elif (pos <neg and rate > 3):
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

def main():
    analyzerSentenceDictComplete = sentiment_analyzer_dict_complete()
    print("\n>>Analyzer based on the dictionary of emotional words with semantic analyse")
    import sys
    for sentence in sys.argv[1:]:
        pos, neg = analyzerSentenceDictComplete.predict_review(sentence)
        print(sentence + "\npositive:" + str(pos) + "," + "negative:" + str(neg)+"\n")

if __name__ == "__main__":
    main()
