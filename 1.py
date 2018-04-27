import glob
import os
import random
from math import log10
import operator
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import matplotlib.pyplot as plt

def NaiveBayes(firstClassDict, secondClassDict, testDoc , firstProb, secondProb, effectiveFeaturesEmamDict, effectiveFeaturesShahDict):
    """
    This function get two dictionary of words and a test,
    and assign a class for the test
    """
    firstClassWordCounts = 0.0
    secondClassWordCounts = 0.0
    firstClassProbability = firstProb
    secondClassProbability = secondProb

    for word in firstClassDict :
        firstClassWordCounts+= firstClassDict[word]

    for word in secondClassDict :
        secondClassWordCounts+= secondClassDict[word]

    for word in testDoc:
        #smoothing +1
        wordCountInFirstClass = 0.0
        wordCountInSecondClass = 0.0
        if word in firstClassDict:
            wordCountInFirstClass = firstClassDict[word] + 1.0
        else:
            wordCountInFirstClass = 1.0
        if word in secondClassDict:
            wordCountInSecondClass = secondClassDict[word] + 1.0
        else :
            wordCountInSecondClass = 1.0

        firstClassProbability += log10( wordCountInFirstClass / (firstClassWordCounts+len(firstClassDict.keys())))
        secondClassProbability += log10( wordCountInSecondClass / (secondClassWordCounts+len(secondClassDict.keys())))

        #find effect of word in emam class
        if word in effectiveFeaturesEmamDict:
            effectiveFeaturesEmamDict[word] += (firstClassProbability-secondClassProbability)
        else:
            effectiveFeaturesEmamDict[word] = firstClassProbability-secondClassProbability

        #find effect of word in shah class
        if word in effectiveFeaturesShahDict:
            effectiveFeaturesShahDict[word] += (secondClassProbability-firstClassProbability)
        else :
            effectiveFeaturesShahDict[word] = (secondClassProbability-firstClassProbability)

    return firstClassProbability,secondClassProbability


def preProcessing(doc):
    """
    This function remove punctuations and some useless prepositions and return a list of words.
    """
    junkList = [".", "-", "]", "[", "،", "؛", ":", " ", ")", "(", "!", "؟", "«", "»", "ْ", " "]
    junkWords = ["که", "از", "با", "برای", "با", "به", "را", "هم", "و", "در", "تا", "یا"]
    result =[]
    for word in doc:
        for char in junkList :
            if char in word :
                word = word.replace(char, "")
        word.strip()
        if word not in junkWords :
            result.append(word)
    return result

def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    print ('Confusion matrix')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

emamPath = '/Users/kiarash/PycharmProjects/NLP_HW3_NaiiveBayse/emam2'
shahPath = '/Users/kiarash/PycharmProjects/NLP_HW3_NaiiveBayse/shah2'

emamAllFiles = glob.glob(os.path.join(emamPath, '*.txt'))
shahAllFiles = glob.glob(os.path.join(shahPath, '*.txt'))
emamList = []
shahList = []

trainTestSplitFactor = 0.1

numberOfEmamTrainSamples = int((1-trainTestSplitFactor) * len(emamAllFiles))
numberOfShahTrainSamples = int((1-trainTestSplitFactor) * len(shahAllFiles))

#randomly put train datas into lists for both emam and shah
emamTrainFiles = random.sample(emamAllFiles, numberOfEmamTrainSamples)
shahTrainFiles = random.sample(shahAllFiles, numberOfShahTrainSamples)

emamTestFiles = []
shahTestFiles = []

#put emam test files names in the emamTestFiles list
for fileName in emamAllFiles :
    if fileName not in emamTrainFiles :
        emamTestFiles.append(fileName)

#put shah test files names in the shahTestFiles list
for fileName in shahAllFiles:
    if fileName not in shahTrainFiles:
        shahTestFiles.append(fileName)

#put emam train files text into a list and preprocess them
for fileName in emamTrainFiles:
    emamList.append(preProcessing(open(fileName,'r').read().split(" ")))

#put shah train files text into a list and preprocess them
for fileName in shahTrainFiles:
    shahList.append(preProcessing(open(fileName,'r').read().split(" ")))

emamDict = {}
shahDict = {}

#put emam data into dictionary
for speech in emamList :
    for word in speech:
        if word in emamDict:
            emamDict[word] += 1
        else:
            emamDict[word] = 1

#put shah data into dictionary
for speech in shahList :
    for word in speech :
        if word in shahDict:
            shahDict[word] += 1
        else:
            shahDict[word] = 1

#probability of being in the first class without words
firstProb = len(emamTrainFiles) / (len(emamTrainFiles) + len(shahTrainFiles))

#probability of being in the second class without words
secondProb = len(shahTrainFiles) / (len(emamTrainFiles) + len(shahTrainFiles))


#use this following two lists for confusion matrix and precision_recall_fscore
trueList = []
predictList = []

#use this following two dictionaries for finding most effective features for each class
effectiveFeaturesEmamDict = {}
effectiveFeaturesShahDict = {}

#Assign a class to emam test files with naive bayes
for fileName in emamTestFiles:
    sentences = open(fileName,'r').read().split(".")
    for sentence in sentences :
        trueList.append(0)
        emamList = preProcessing(sentence.split(" "))
        result = NaiveBayes(emamDict, shahDict, emamList, firstProb, secondProb, effectiveFeaturesEmamDict, effectiveFeaturesShahDict)
        if result[0] > result[1]:
            predictList.append(0)
        else:
            predictList.append(1)

#Assign a class to shah test files with naive bayes
for fileName in shahTestFiles:
    sentences = open(fileName, 'r').read().split(".")
    for sentence in sentences:
        trueList.append(1)
        shahList = preProcessing(sentence.split(" "))
        result =  NaiveBayes(emamDict, shahDict, shahList , firstProb, secondProb, effectiveFeaturesEmamDict, effectiveFeaturesShahDict)
        if result[0] > result[1] :
            predictList.append(0)
        else :
            predictList.append(1)

#sort the features by effect
effectiveFeaturesEmam = sorted(effectiveFeaturesEmamDict.items(), key=operator.itemgetter(1),reverse=True)
effectiveFeaturesShah = sorted(effectiveFeaturesShahDict.items(), key=operator.itemgetter(1),reverse=True)
print (effectiveFeaturesShah[0:5],"\n")
print (effectiveFeaturesEmam[0:5],"\n")

#calculate precision, recall, fscore
precision,recall,fscore,support = precision_recall_fscore_support(trueList, predictList)

#printing precision, recall, fscore for both emam
print ("number of emam test sentences -> ",support[0])
print ("emam precision -> ",precision[0])
print ("emam recall -> ",recall[0])
print ("emam fscore -> ",fscore[0],"\n\n")

#printing precision, recall, fscore for both shah
print ("number of shah test sentences -> ",support[1])
print ("shah precision -> ",precision[1])
print ("shah recall -> ",recall[1])
print ("shah fscore -> ",fscore[1],"\n\n")


#draw confusion matrix plot
confusionMatrix = confusion_matrix(trueList, predictList)
plt.figure()
plot_confusion_matrix(confusionMatrix, classes=['emam','shah'])
plt.show()