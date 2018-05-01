import glob
import os
import hazm
import random
def preProcessing (doc):
    junkList = [".", "-", "]", "[", "،", "؛", ":", ")", "(", "!", "؟", "»", "«", "ْ"]
    junkWords = ["که", "از", "با", "برای", "با", "به", "را", "هم", "و", "در", "تا", "یا", "هر", "می", "بر"]
    pronouns = ["من", "تو", "او", "ما", "شما", "ایشان", "آن‌ها", "این‌ها", "آن", "این", "اونجا", "آنجا", "انجا",
                "اینها", "آنها"
        , "اینکه"]
    for char in junkList:
        doc = doc.replace(char, "")
    doc.strip()
    doc = hazm.Normalizer().normalize(doc)
    return doc

trainTestSplitFactor = 0.1

emamPath = '/Users/kiarash/PycharmProjects/NLP_HW3_NaiiveBayse/emam2'
shahPath = '/Users/kiarash/PycharmProjects/NLP_HW3_NaiiveBayse/shah2'

emamAllFiles = glob.glob(os.path.join(emamPath, '*.txt'))
shahAllFiles = glob.glob(os.path.join(shahPath, '*.txt'))

numberOfEmamTrainSamples = int((1-trainTestSplitFactor) * len(emamAllFiles))
numberOfShahTrainSamples = int((1-trainTestSplitFactor) * len(shahAllFiles))

# randomly put train datas into lists for both emam and shah
emamTrainFiles = random.sample(emamAllFiles, numberOfEmamTrainSamples)
shahTrainFiles = random.sample(shahAllFiles, numberOfShahTrainSamples)

emamCounter = 0
shahCounter = 0

emamTrainSentences =[]
shahTrainSentences =[]

emamTestFiles = []
shahTestFiles = []

emamTestSentences = []
shahTestSentences = []
fileTrain = open("Train.txt","w")
fileTest = open("Test.txt","w")

for fileName in emamTrainFiles:
    sentences = hazm.sent_tokenize(open(fileName, 'r').read())
    for s in sentences:
        emamTrainSentences.append(s)
for fileName in shahTrainFiles:
    sentences = hazm.sent_tokenize(open(fileName, 'r').read())
    for s in sentences:
        shahTrainSentences.append(s)

#put emam test files names in the emamTestFiles list
for fileName in emamAllFiles :
    if fileName not in emamTrainFiles :
        emamTestFiles.append(fileName)
#put shah test files names in the shahTestFiles list
for fileName in shahAllFiles:
    if fileName not in shahTrainFiles:
        shahTestFiles.append(fileName)

for fileName in emamTestFiles:
    sentences = hazm.sent_tokenize(open(fileName, 'r').read())
    for s in sentences:
        emamTestSentences.append(s)
for fileName in shahTestFiles:
    sentences = hazm.sent_tokenize(open(fileName, 'r').read())
    for s in sentences:
        shahTestSentences.append(s)

print (len(emamTrainSentences),len(shahTrainSentences))

while emamCounter <len(emamTrainSentences) and shahCounter <len(shahTrainSentences):
    emamTrainSentences[emamCounter]= preProcessing(emamTrainSentences[emamCounter])
    shahTrainSentences[shahCounter]= preProcessing(shahTrainSentences[shahCounter])
    if len(shahTrainSentences[shahCounter]) > 0 and len(emamTrainSentences[emamCounter]) > 0:
        fileTrain.write("1 |"+emamTrainSentences[emamCounter]+"\n")
        fileTrain.write("0 |"+shahTrainSentences[shahCounter]+"\n")
    emamCounter+=1
    shahCounter+=1

emamCounter=0
shahCounter=0

while emamCounter <len(emamTestSentences) and shahCounter <len(shahTestSentences):
    emamTestSentences[emamCounter]= preProcessing(emamTestSentences[emamCounter])
    shahTestSentences[shahCounter]= preProcessing(shahTestSentences[shahCounter])
    if len(shahTestSentences[shahCounter]) > 0 and len(emamTestSentences[emamCounter]) > 0:
        fileTest.write("1 |"+emamTestSentences[emamCounter]+"\n")
        fileTest.write("0 |"+shahTestSentences[shahCounter]+"\n")
    emamCounter+=1
    shahCounter+=1