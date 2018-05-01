import glob
import os
import random
from math import log10
import operator
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import matplotlib.pyplot as plt
import hazm

class NaiveBayesClassifier():
    
    def __init__(self, firstClassDataPath , secondClassDataPath , trainTestSplitFactor =0.1 , firstClassLabel = "1" ,
                 secondClassLabel = "2", flagShowPlot = True , effectiveFeaturesNumber = 10 , flagPrintResults = True):

        self.firstClassLabel = firstClassLabel
        self.secondClassLabel = secondClassLabel

        self.firstClassDataPath = firstClassDataPath     
        self.secondClassDataPath = secondClassDataPath
        self.trainTestSplitFactor = trainTestSplitFactor

        self.getFiles()
        self.splitTrainTest()
        self.readTrainTestFiles()
        self.buildClassesDictionary()
        self.runClassifier()
        self.evaluatePRFS_CM_Acc()
        self.sortEffectiveFeatures()

        if flagPrintResults:
            self.printResults(effectiveFeaturesNumber)
        if flagShowPlot:
            plt.figure()
            self.plot_confusion_matrix(classes=['emam','shah'])
            plt.show()
            plt.close()

    def getFiles(self):
        """This function find all the file names in the path directory of each class"""
        self.firstClassAllFiles = glob.glob(os.path.join(self.firstClassDataPath, '*.txt'))
        self.secondClassAllFiles = glob.glob(os.path.join(self.secondClassDataPath, '*.txt'))
    
    def splitTrainTest(self):
        """This function split the data into train & test by splitFactor"""
        numberOf1stTrainSamples = int((1 - self.trainTestSplitFactor) * len(self.firstClassAllFiles))
        numberOf2stTrainSamples = int((1 - self.trainTestSplitFactor) * len(self.secondClassAllFiles))

        # randomly put train files name into lists
        self.firstClassTrainFiles = random.sample(self.firstClassAllFiles, numberOf1stTrainSamples)
        self.secondClassTrainFiles = random.sample(self.secondClassAllFiles, numberOf2stTrainSamples)

        self.firstClassTestFiles = []
        self.secondClassTestFiles= []

        #put test files name into lists
        for fileName in self.firstClassAllFiles:
            if fileName not in self.firstClassTrainFiles:
                self.firstClassTestFiles.append(fileName)


        for fileName in self.secondClassAllFiles:
            if fileName not in self.secondClassTrainFiles:
                self.secondClassTestFiles.append(fileName)
                
    def readTrainTestFiles(self):
        """This function loads all train and test datas into list and preprocess them"""
        self.firstClassTrainList = []
        self.secondClassTrainList = []
        self.firstClassTestList = []
        self.secondClassTestList = [] 
        
        for fileName in self.firstClassTrainFiles:
            self.firstClassTrainList.append(self.preProcessing(open(fileName, 'r').read()))

        for fileName in self.secondClassTrainFiles:
            self.secondClassTrainList.append(self.preProcessing(open(fileName, 'r').read()))

        for fileName in self.firstClassTestFiles:
            sentences = hazm.sent_tokenize(open(fileName, 'r').read())
            for sentence in sentences:
                self.firstClassTestList.append(self.preProcessing(sentence))
        
        for fileName in self.secondClassTestFiles:
            sentences = hazm.sent_tokenize(open(fileName, 'r').read())
            for sentence in sentences:
                self.secondClassTestList.append(self.preProcessing(sentence))

    def buildClassesDictionary(self):
        """This function build a dictionary of words for each class"""
        self.firstClassTrainDictionary = {}
        self.secondClassTrainDictionary = {}
        
        for speech in self.firstClassTrainList:
            for word in speech:
                if word in self.firstClassTrainDictionary:
                    self.firstClassTrainDictionary[word] += 1
                else:
                    self.firstClassTrainDictionary[word] = 1
                    
        for speech in self.secondClassTrainList:
            for word in speech:
                if word in self.secondClassTrainDictionary:
                    self.secondClassTrainDictionary[word] += 1
                else:
                    self.secondClassTrainDictionary[word] = 1
                    
    def runClassifier(self):
        """This function send test data to classifier and build a predection list"""
        self.predictList = []
        self.trueList = []
        self.effectiveFeaturesFirstDict = {}
        self.effectiveFeaturesSecondDict = {}

        firstClassProb = len(self.firstClassTrainFiles) / (len(self.firstClassTrainFiles) + len(self.secondClassTrainFiles))
        secondClassProb = len(self.secondClassTrainFiles) / (len(self.firstClassTrainFiles) + len(self.secondClassTrainFiles))

        for data in self.firstClassTestList :
            self.trueList.append(0)
            result = self.NaiveBayes(data , firstClassProb , secondClassProb)
            if result[0] > result[1]:
                self.predictList.append(0)
            else:
                self.predictList.append(1)

        for data in self.secondClassTestList :
            self.trueList.append(1)
            result = self.NaiveBayes(data , firstClassProb , secondClassProb)
            if result[0] > result[1]:
                self.predictList.append(0)
            else:
                self.predictList.append(1)
                
    def NaiveBayes(self, testDoc, firstProb, secondProb):
        """This function get a test and classify it by naive bayes algorithm"""
        firstClassWordCounts = 0.0
        secondClassWordCounts = 0.0
        firstClassProbability = firstProb
        secondClassProbability = secondProb
        for word in self.firstClassTrainDictionary:
            firstClassWordCounts += self.firstClassTrainDictionary[word]

        for word in self.secondClassTrainDictionary:
            secondClassWordCounts += self.secondClassTrainDictionary[word]

        for word in testDoc:
            # smoothing +1
            wordCountInFirstClass = 0.0
            wordCountInSecondClass = 0.0
            if word in self.firstClassTrainDictionary:
                wordCountInFirstClass = self.firstClassTrainDictionary[word] + 1.0
            else:
                wordCountInFirstClass = 1.0
            if word in self.secondClassTrainDictionary:
                wordCountInSecondClass = self.secondClassTrainDictionary[word] + 1.0
            else:
                wordCountInSecondClass = 1.0

            firstClassProbability += log10(wordCountInFirstClass / (firstClassWordCounts + len(self.firstClassTrainDictionary.keys())))
            secondClassProbability += log10(wordCountInSecondClass / (secondClassWordCounts + len(self.secondClassTrainDictionary.keys())))

            # find effect of word in 1st class
            if word in self.effectiveFeaturesFirstDict:
                self.effectiveFeaturesFirstDict[word] += (firstClassProbability - secondClassProbability)
            else:
                self.effectiveFeaturesFirstDict[word] = firstClassProbability - secondClassProbability

            # find effect of word in 2st class
            if word in self.effectiveFeaturesSecondDict:
                self.effectiveFeaturesSecondDict[word] += (secondClassProbability - firstClassProbability)
            else:
                self.effectiveFeaturesSecondDict[word] = (secondClassProbability - firstClassProbability)

        return firstClassProbability, secondClassProbability
    
    def preProcessing(self , doc, level=0):
        """
        This function remove punctuations and some useless prepositions and return a list of words.
        """
        junkList = [".", "-", "]", "[", "،", "؛", ":", ")", "(", "!", "؟", "»", "«", "ْ"]
        junkWords = ["که", "از", "با", "برای", "با", "به", "را", "هم", "و", "در", "تا", "یا", "هر", "می", "بر"]
        pronouns = ["من", "تو", "او", "ما", "شما", "ایشان", "آن‌ها", "این‌ها", "آن", "این", "اونجا", "آنجا", "انجا",
                    "اینها", "آنها", "اینکه"]
        for char in junkList:
            doc = doc.replace(char, " ")
        result = []
        doc = hazm.Normalizer().normalize(doc)
        doc = hazm.word_tokenize(doc)
        for word in doc:
            word.strip()
            if word not in junkWords and word not in pronouns:
                result.append(word)
        return result

    def evaluatePRFS_CM_Acc(self):
        """This function calculate precision, recall, fscore, support , accuracy , confusion matrix"""
        self.precision, self.recall, self.fscore, self.support = precision_recall_fscore_support(self.trueList, self.predictList)
        self.confusion_matrix = confusion_matrix(self.trueList, self.predictList)
        self.accuracy = (self.confusion_matrix[0][0] + self.confusion_matrix[1][1]) / (self.support[0] + self.support[1])

    def plot_confusion_matrix(self, classes , normalize=False ,title = "confusion matrix" , cmap=plt.cm.Blues ):
        """This function make a plot of confusion matrix"""
        print('Confusion matrix')
        print(self.confusion_matrix)

        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = self.confusion_matrix.max() / 2.
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            plt.text(j, i, format(self.confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if self.confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def sortEffectiveFeatures(self):
        """This function sort the effective features of each class by their effect"""
        self.effectiveFeatures1stClass = sorted(self.effectiveFeaturesFirstDict.items(), key=operator.itemgetter(1),
                                                reverse=True)
        self.effectiveFeatures2stClass = sorted(self.effectiveFeaturesSecondDict.items(), key=operator.itemgetter(1),
                                                reverse=True)

    def printResults(self , n):
        """This function prints n most effective features of each class and precision, recall, fscore , accuracy"""
        print(self.firstClassLabel,"effective features: ", [y[0] for y in self.effectiveFeatures1stClass[0:n]], "\n")
        print(self.secondClassLabel,"effective features: ", [y[0] for y in self.effectiveFeatures2stClass[0:n]], "\n")

        print("number of",self.firstClassLabel,"test sentences -> ", self.support[0])
        print(self.firstClassLabel,"precision -> ", self.precision[0])
        print(self.firstClassLabel,"recall -> ", self.recall[0])
        print(self.firstClassLabel,"fscore -> ", self.fscore[0], "\n\n")

        print("number of",self.secondClassLabel,"test sentences -> ", self.support[1])
        print(self.secondClassLabel,"precision -> ", self.precision[1])
        print(self.secondClassLabel,"recall -> ", self.recall[1])
        print(self.secondClassLabel,"fscore -> ", self.fscore[1], "\n\n")

        print("Accuracy :(number of true predicts/ total predictions) = ", self.accuracy)

if __name__ == '__main__':
    emamPath = '/Users/kiarash/PycharmProjects/NLP_HW3_NaiiveBayse/emam2'
    shahPath = '/Users/kiarash/PycharmProjects/NLP_HW3_NaiiveBayse/shah2'
    x = NaiveBayesClassifier(emamPath , shahPath , trainTestSplitFactor =0.1 , firstClassLabel = "emam" ,
                 secondClassLabel = "shah", flagShowPlot = True , effectiveFeaturesNumber = 10 , flagPrintResults=True)

