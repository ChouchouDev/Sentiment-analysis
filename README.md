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
sentences = [
    "Good and not good",  # 1. neutral sentence, used in the all sentence below for the reason of a good visual comparison
    "This camera is at a good price and easy to use. Good and not good",  #2. positive sentence
    "This camera is bad and difficult to use.  Good and not good",         # 3. negative sentence
    "This camera isn't at a good price and easy to use. Good and not good",  # 4. sentence of negation
    "This camera is never not at a good price and easy to use. Good and not good", # 5. sentence of double negation
    "This camera isn't at a good price but easy to use! Good and not good",  # 6. conjuction "but"
    "This camera is at a good price and easy to use! Good and not good",  # 7. punctuation emphasis
    "This camera is at a very good price and easy to use. Good and not good",  # 8. degree word
]
```
__Analyzer based on the classification algo Bayes__

|  |  | positive | negative |    
| - | - |- | -- |    
| 1 | Good and not good | 0.585644 | 0.4143556 |  
|2|This camera is at a good price and easy to use. Good and not good|0.8217495|0.178250|  
|3|This camera is bad and difficult to use.  Good and not good|0.31580473|0.68419|  
|4|This camera isn't at a good price and easy to use. Good and not good|0.7748391|0.225160|  
|5|This camera is never not at a good price and easy to use. Good and not good|0.7376330|0.262366|  
|6|This camera isn't at a good price but easy to use! Good and not good| 0.8109000|0.189099|  
|7|This camera is at a good price and easy to use! Good and not good|0.8719950|0.128004|  
|8|This camera is at a very good price and easy to use. Good and not good|0.8398022|0.160197|  

__Analyzer based on the dictionary of emotional words with semantic analysis__

|  |  | positive | negative |    
| - | - |- | -- |    
|1|Good and not good|0.5|0.5|  
|2|This camera is at a good price and easy to use. Good and not good|0.698292|0.3017077|  
|3|This camera is bad and difficult to use.  Good and not good|0.246305|0.753694|  
|4|This camera isn't at a good price and easy to use. Good and not good|0.3017077|0.69829|  
|5|This camera is never not at a good price and easy to use. Good and not good|0.698292|0.301707|  
|6|This camera isn't at a good price but easy to use! Good and not good|0.375329|0.624670|  
|7|This camera is at a good price and easy to use! Good and not good|0.749589|0.250410|  
|8|This camera is at a very good price and easy to use. Good and not good|0.834173|0.165826|  

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
