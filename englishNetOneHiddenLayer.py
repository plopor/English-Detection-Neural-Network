import numpy as np
import random
import copy

#constructor for a hidden layer neuron
class neuron ():
    def __init__(self, inputLayer, weights):
        self.output = 0
        for x in range (len(inputLayer)):
            self.output += (float(inputLayer[x]) * weights[x])
        self.output = 1 / (1 + np.exp(-self.output))
        self.inputLayer = inputLayer
        self.weights = weights

#constructor for output neuron having been given hidden layer as the input layer
class outNeuron ():
    def __init__(self, hiddenLayer, weights):
        self.output = 0
        for x in range (len(hiddenLayer)):
            self.output += (hiddenLayer[x].output * weights[x])
        self.output = 1 / (1 + np.exp(-self.output))
        self.hiddenLayer = hiddenLayer
        self.weights = weights

maxLength = 15 # alter max length of input words
engLib = open("english.txt", "r")
answers = open("answers.txt", "r")

#takes a having been converted text file, list of words and turns each word into a series of 0 and 1,
#each letter being 26 0s or 1s depending on the letter. The list is 26 * (max amount of letters per word) in length
def convertToVector(los, maxLength):
    vecList = []
    for word in los.split(): #make a new word vector at a space or newline
        vec = ''
        n = len(word) #find length of current word
        for i in range(n): #for each character in current word
            charNum = ord(word[i])-97 #takes the unicode of current character and arranges it from 0 (a is 0, b is 1 etc)
            value = (str(0) * charNum) + str(1) + str(0) * (25 - charNum) #in a list of 25 0s, the 1 is placed at its correspinding letter's location
            vec = vec + value
        if n < maxLength:
            vec = vec + str(0) * 26 * (maxLength - n) #fills in the rest of the unused letter spaces for words shorter than the max with 0s
        vecList.append(vec)
    print("Number of training cases:")
    print(len(vecList))
    print("Training cases:")
    print(vecList)
    return vecList

#makes a layer of hidden neurons
def makeLayer(inputLayer, weights, neuronNum):
    outList = []
    for x in range (neuronNum): #makes a specified amount of hidden neurons
        outList.append(neuron(inputLayer, weights[x]))
    return outList

#create a random set of weights for initial use to be altered by training
def initWeights(inputsFromLayer, numberOfNeurons):
    neurons = []
    for y in range (numberOfNeurons): #create a set of weights for each hidden layer neuron
        weights = []
        for x in range (inputsFromLayer): #create random weight values for each incoming value
            weights.append(random.uniform(-1, 1))
        neurons.append(weights)
    return neurons

#back prop for output neuron
def backPropOut(ideal, outputNeuron):
    error = -2*(ideal - outputNeuron.output) #calculate error
    adjust = error * outputNeuron.output * (1 - outputNeuron.output) #calculate adjustment factor
    for x in range (len(outputNeuron.hiddenLayer)): #apply changes to weight for each incoming weight to output neuron
        outputNeuron.weights[x] -= outputNeuron.hiddenLayer[x].output * adjust
    return adjust

#back propogation for the hidden layer
def backPropInput(passBack, saveWeights, outLayerProp):
    for x in range (len (outLayerProp)): #apply algorithm for each neuron in hidden layer
        temp = []
        adjust = passBack * saveWeights[x] * outLayerProp[x].output * (1 - outLayerProp[x].output) #adjustment to make on current neuron
        for y in range (len (outLayerProp[x].inputLayer)): #apply algorithm for each incoming weight in current neuron
            temp.append(outLayerProp[x].weights[y] - float(outLayerProp[x].inputLayer[y]) * adjust)
        outLayerProp[x].weights = temp

def prediction (train, trainOut, averageWeights, averageOut):
    # displays the percentage of test data guessed correctly
    correct = 0
    print("accuracy in guessing training data:")
    for examples in range(len(train)):
        guessLayer = makeLayer(train[examples], averageWeights, 10)
        guess = outNeuron(guessLayer, averageOut)
        if (int(round(guess.output)) == int(trainOut[examples])):
            correct = correct + 1
    print(correct / len(train) * 100)

def training (iterations, learnRate, repeats, testWeights, testOutWeights, averageWeights, averageOut):
    # initialize some inputs
    train = convertToVector(engLib.read(), 15)  # write the txt file into an array
    trainOut = answers.read()  # write answers that correspond to txt file into array
    # init end weights
    for i in range(10):
        zeroes = [0] * 390
        averageWeights.append(zeroes)
        averageOut.append(0)

    # init layers
    outputs = []
    outputNeuron = None

    # the training process
    for repeat in range(repeats):
        print("EPOCH NUMBER: ")
        print(repeat)
        print("Words processed:")
        for word in range(len(train)):  # loops through each word in the sample list
            print("-------------------------------------------------------------")
            print(len(testWeights[0]))
            print(len(train[word]))

            for iterate in range(iterations):  # iterates each word an amount of time specified by iterations
                # the forwards feed
                outputs = makeLayer(train[word], testWeights, 10)  # make the hidden layer of 10 neurons

                outputNeuron = outNeuron(outputs, testOutWeights[0])  # make the output neuron, contains the guess as well

                # print(outputNeuron.output)

                # the backwards pass
                saveWeights = copy.deepcopy(outputNeuron.weights)  # keep a copy of old weights to use for hidden layer propogation
                passBack = backPropOut(float(trainOut[word]), outputNeuron)  # back-propogate through the output neuron, weights are updated
                backPropInput(passBack, saveWeights, outputs)  # back-propogate through the hidden layer, weights are updated

                # updating input weights for use in next iteration
                testWeights = []
                for update in range(len(outputs)):
                    testWeights.append(outputs[update].weights)
                testOutWeights = [outputNeuron.weights]

            # print("altered hidden layer weights:") #prints out the hidden layer's weights for current word
            # print(outputs[0].weights)
            # print(outputs[1].weights)
            # print(outputs[2].weights)
            # print(outputs[3].weights)
            # print(outputs[4].weights)
            # print(outputs[5].weights)
            # print(outputs[6].weights)
            # print(outputs[7].weights)
            # print(outputs[8].weights)
            # print(outputs[9].weights)
            # print("altered output weights:") #prints out the output neuron's weights for current word
            # print(outputNeuron.weights)
            print(word)

            # constructs a sum of all the weights for the words in the bank
            for neurons in range(len(outputs)):
                for weights in range(len(outputs[neurons].weights)):
                    averageWeights[neurons][weights] += outputs[neurons].weights[weights]  # adds current hidden layer weights to the sum of previous weights
            for outWeights in range(len(outputNeuron.weights)):
                averageOut[outWeights] += outputNeuron.weights[outWeights]  # adds current output neuron weights to the sum of previous weights

    print("summed hidden layer weights:")  # display resulted summed weights for the hidden layer after training
    print(averageWeights[0])
    print(averageWeights[1])
    print(averageWeights[2])
    print(averageWeights[3])
    print(averageWeights[4])
    print(averageWeights[5])
    print(averageWeights[6])
    print(averageWeights[7])
    print(averageWeights[8])
    print(averageWeights[9])
    print("summed output weights:")  # display resulted summed weights for output neuron post training
    print(averageOut)

    # averages each weight by total amount of test words
    for neurons in range(len(averageWeights)):
        for weights in range(len(averageWeights[neurons])):
            averageWeights[neurons][weights] = learnRate * averageWeights[neurons][weights] / (len(train) * repeats)  # for hidden layer
    for outWeights in range(len(averageOut)):
        averageOut[outWeights] = learnRate * averageOut[outWeights] / (len(train) * repeats)  # for output neuron

    prediction (train, trainOut, averageWeights, averageOut)

    # display resulted averaged weights
    print("averaged hidden layer weights:")  # for hidden layer
    print(averageWeights[0])
    print(averageWeights[1])
    print(averageWeights[2])
    print(averageWeights[3])
    print(averageWeights[4])
    print(averageWeights[5])
    print(averageWeights[6])
    print(averageWeights[7])
    print(averageWeights[8])
    print(averageWeights[9])
    print("averaged output weights:")  # for output neuron
    print(averageOut)

if __name__ == '__main__':
    loadWeights = input("Load weights? y/n: ")

    # init weight holders
    averageWeights = []
    averageOut = []

    # train fresh
    if (loadWeights == "n"):

        iterations = 1  # set number of iterations
        # initialize base average weights for hidden, then output layers
        learnRate = 1  # set learning rate, play around to alter results
        repeats = 400  # set amount of times to repeat on the training set (epochs)

        testWeights = initWeights(390, 10)  # change word length as fit
        # print(testWeights[0])
        testOutWeights = initWeights(10, 1)
        # print (testOutWeights)
        training(iterations, learnRate, repeats, testWeights, testOutWeights, averageWeights, averageOut)

    # load preexisting weights
    else:
        weightFile = open("layerOneWeights.txt", "r").read()
        loadWeights = []
        for weightSet in weightFile.split("\n"):
            wSet = []
            for weights in weightSet.split():
                wSet.append(float(weights))
            loadWeights.append(wSet)
        loadOut = []
        outWeightFile = open("outLayerWeights.txt", "r").read()
        for weights in outWeightFile.split():
            loadOut.append(float(weights))

        #display resulted averaged weights
        print("averaged hidden layer weights:") #for hidden layer
        print(loadWeights[0])
        print(loadWeights[1])
        print(loadWeights[2])
        print(loadWeights[3])
        print(loadWeights[4])
        print(loadWeights[5])
        print(loadWeights[6])
        print(loadWeights[7])
        print(loadWeights[8])
        print(loadWeights[9])
        print("averaged output weights:") #for output neuron
        print(loadOut)

        predict = input("Make prediction? y/n: ")
        if (predict == "y"):
            toPredict = input("Read data from which file? ")
            truth = input("Check against which file? ")
            data = open(toPredict, "r")
            answer = open(truth, "r")
            predictExamples = convertToVector(data.read(), 15)
            prediction(predictExamples, answer.read(), loadWeights, loadOut)

        keepTraining = input("Continue training? y/n: ")
        if (keepTraining == "y"):
            iterations = int(input("Iterations: "))
            learnRate = int(input("Learning rate: "))
            repeats = int(input("Epochs: "))
            training(iterations, learnRate, repeats, loadWeights, [loadOut], averageWeights, averageOut)

    #prompt user to give the net words to guess
    while True:
        word = input("Word: ")
        if (word == "SW"):
            weightFile = open("layerOneWeights.txt", "w+")
            for i in range (len (averageWeights)):
                for j in range (len (averageWeights[i])):
                    weightFile.write(str(averageWeights[i][j]))
                    weightFile.write(" ")
                weightFile.write("\n")
            weightFile.close()
            outWeightFile = open ("outLayerWeights.txt", "w+")
            for i in range (len (averageOut)):
                outWeightFile.write(str(averageOut[i]))
                outWeightFile.write(" ")
            outWeightFile.close()
            continue
        word = convertToVector(word, maxLength)
        guessLayer = makeLayer (word[0], averageWeights, 10)
        guess = outNeuron (guessLayer, averageOut)
        print(guess.output)

# SUCCESS! NOTE, BY INCREASING THE EPOCHS OF THE NEURAL NET AND DECREASING THE AMOUNT OF ITERATIONS,
# THE ACCURACY IN GUESSING HAS GONE UP BY 30% FROM 65% TO 95%. I WILL KEEP THIS IN MIND FOR FUTURE IMPLEMENTATIONS

# increasing training data -> lower accuracy
# 2000 with 200 -> 98.75% (5 learnRate) 99.55% (1 learnRate)
    # 2000 with 100 -> 95.75% (5 learnRate) 96.8 (1 learnRate)
    # 2000 with 400 -> 99.25%
# 3000 with 200 -> 94%
# 7000 with 200 -> 77%
    # 7000 with 100 -> 76.94% (5 learnRate) 75.5% (1 learnRate)
    # 7000 with 400 -> 80%
    # additional 400 on top of last 7000 -> 83.8%

# NEW 7000 training set, less obscure words
    # 400 -> 86% + 400 -> 88.6%

    

    

