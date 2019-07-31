# English-Detection-Neural-Network
Made in python, this project is a neural network consisting of an input layer of 390 binary inputs, a hidden layer of 10 perceptrons, and one output perceptron that uses back-propagation to determine whether the input word is English or not. Currently being trained with a testing set of 2000 English and non-English entries of maximum 15 letters long, the neural net is able to build its synapse weights accordingly as it learns to guess the aspects that distinguish a valid word from gibberish. By increasing the testing set and iteration amount in future updates, the network will be able to improve the accuracy of its guesses.

As of July 28, 2018, the current implementation limits the training speed, and the training set of 7000 is insufficient in showing any more than 55-56% accuracy in testing data. Significant improvements can be seen however from the increase from the 2000 training data to the 7000, and likewise for a longer training time. I will implement a similar project using Keras and Tensorflow and compare the results. This will remain on the backburner.

Some results:

 increasing training data -> lower accuracy
 2000 with 200 -> 98.75% (5 learnRate) 99.55% (1 learnRate)
     2000 with 100 -> 95.75% (5 learnRate) 96.8 (1 learnRate)
     2000 with 400 -> 99.25%
 3000 with 200 -> 94%
 7000 with 200 -> 77%
     7000 with 100 -> 76.94% (5 learnRate) 75.5% (1 learnRate)
     7000 with 400 -> 80%
     additional 400 on top of the previous 80% -> 83.8%

 NEW 7000 training set, less obscure words
     400 -> 86% + 400 -> 88.6%
