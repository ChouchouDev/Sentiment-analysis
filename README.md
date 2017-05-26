# Sentiment-analysis
## Abstract
Emotional analysis (SA), also known as the tendency analysis and opinion mining, is the process of analyzing, processing, summarizing and reasoning the subjective text with emotional tendency. This project aim at develop a system of sentiment analysis for a product or a service. In this project, we have taken two approaches to create sentiment analyzer. One is based on the dictionary of positive and negative words, with the supplementary analysis: negation analysis, degree adverb addition, etc. The other one is based on the classification algorithm Naive Bayes, machine learning domain. The main task of the project is the first approach, but the compare of two approaches are also meaningful.

## Description
>**Sentiment analysis** (sometimes known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.  
  
>Generally speaking, sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or the overall contextual polarity or emotional reaction to a document, interaction, or event. The attitude may be a judgment or evaluation (see appraisal theory), affective state (that is to say, the emotional state of the author or speaker), or the intended emotional communication (that is to say, the emotional effect intended by the author or interlocutor).

## Contributor
* Miaobing CHEN
* Rui SUN

## Stuff of developpement
* Python 3.x
* NLTK

## Usage
Use analyzer based on the classification algo Bayes
```
>python sentiment_analyzer_dict_complete.py "Sentence for analyze"
```
Use analyzer based on the dictionary of emotional words with semantic analyse
```
>python sentiment_analyzer_classification.py "Sentence for analyze"
```
We can analyze several sentences seperated by space at the same time.
```
>>>>>Preparation TEST
dataset of reviews{'1': 2000, '2': 2000, '3': 0, '4': 2000, '5': 2000}
===============Performance TEST==============

>>Analyzer based on the classification algo Bayes
TP:498
FP:33
TN:542
FN:127
Accuracy:0.866667

>>Analyzer based on the dictionary of emotional words with semantic analysis
TP:535
FP:227
TN:348
FN:90
Accuracy:0.735833
===============Precise TEST==============

>>Analyzer based on the classification algo Bayes
Good and not good
positive:0.585644330224252,negative:0.41435566977574795
This camera is at a good price and easy to use. Good and not good
positive:0.8217495629278319,negative:0.17825043707216814
This camera is bad and difficult to use.  Good and not good
positive:0.31580473181269547,negative:0.6841952681873045
This camera isn't at a good price and easy to use. Good and not good
positive:0.7748391586001706,negative:0.22516084139982934
This camera is never not at a good price and easy to use. Good and not good
positive:0.7376330038493273,negative:0.26236699615067277
This camera isn't at a good price but easy to use! Good and not good
positive:0.8109000267896666,negative:0.18909997321033342
This camera is at a good price and easy to use! Good and not good
positive:0.8719950323508162,negative:0.12800496764918373
This camera is at a very good price and easy to use. Good and not good
positive:0.8398022979799675,negative:0.16019770202003245

>>Analyzer based on the dictionary of emotional words with semantic analysis
Good and not good
positive:0.5,negative:0.5
This camera is at a good price and easy to use. Good and not good
positive:0.6982922120613468,negative:0.30170778793865316
This camera is bad and difficult to use.  Good and not good
positive:0.2463050323939432,negative:0.7536949676060569
This camera isn't at a good price and easy to use. Good and not good
positive:0.30170778793865316,negative:0.6982922120613468
This camera is never not at a good price and easy to use. Good and not good
positive:0.6982922120613468,negative:0.30170778793865316
This camera isn't at a good price but easy to use! Good and not good
positive:0.3753293607126822,negative:0.6246706392873178
This camera is at a good price and easy to use! Good and not good
positive:0.7495899177863296,negative:0.25041008221367045
This camera is at a very good price and easy to use. Good and not good
positive:0.8341735691605251,negative:0.1658264308394749
```
## Resources Descriptions
* **Test.py**  
  Test for two analyzer 
  
* **sentiment_analyzer_classification.py**   
  Analyzer based on the classification algo Bayes
  
* **sentiment_analyzer_dict_complete.py**  
  Analyzer based on the dictionary of emotional words with semantic analyse
  
* **SentiWordNetReader.py**  
  Source code for reading SentiWordNet and write the file **SentiWordNet_simple_format.txt**
  
* **SentiWordNet_simple_format.txt**  
  Lexicon in a simple format generated by SentiWordNet
  
* **degree.txt**  
  A list of adverb of degree, collected from http://en.wiktionary.org/wiki/Category:English_degree_adverbs
  
* **negation.txt**  
  A lsit of words or expression that means negation of sentence

## Reference
* https://en.wikipedia.org/wiki/Sentiment_analysis
* https://en.wikipedia.org/wiki/Natural_language_processing
* Dataset: http://times.cs.uiuc.edu/~wang296/Data/
* SentiWordNet: http://sentiwordnet.isti.cnr.it/
