import numpy as np
import random
import math

# loads test data in format given in the homeworks (first column is ID, rest is properties).
# returns training ID list, training properties list
# code is from HW1
def loadTestDataHW(filepath, doDebugPrint = False):
    # code provided from prof. in HW1 for loading test data
    data = np.loadtxt(filepath, delimiter=',')
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    if doDebugPrint:
        print('array of labels: shape ' + str(np.shape(labels)))
        print('array of feature matrix: shape ' + str(np.shape(features)))
    return labels, features

# splits a single set into a validation and a training set. returns all four
# percentage to sample is the amount of the set to be placed in the validation set instead of training
# ex. 20 is 20%, 40 is 40%, etc. percentageToSample must be <= 50
def splitValidationTraining(labels, features, percentageToSample, doDebugPrint = False):
    if percentageToSample > 50:
        raise Exception("percentageToSample must be 50% or lower")

    # new sets
    validationSetIndices = []
    validationSetLabels = []
    validationSetFeatures = []
    validationSetIndices = []
    trainingSetLabels = labels.tolist()
    trainingSetFeatures = features.tolist()

    # generate indexes to use for validation set
    VALIDATION_SAMPLE_SIZE = math.trunc(len(features) * (percentageToSample / 100)) # % of original data
    for i in range(VALIDATION_SAMPLE_SIZE):
        randIndex = random.randrange(VALIDATION_SAMPLE_SIZE)
        while (randIndex in validationSetIndices):
            randIndex = random.randrange(VALIDATION_SAMPLE_SIZE)
        validationSetIndices.append(randIndex)

    # move validation set elements from data to validationSet
    for i in range(len(validationSetIndices)):
        validationSetLabels.append(trainingSetLabels.pop(i))
        validationSetFeatures.append(trainingSetFeatures.pop(i))
    
    # print data
    if doDebugPrint:
        print("data size\n" + str(len(features)),"\n")
        print("validation set size\n" + str(len(validationSetFeatures)))
        print(str(len(validationSetLabels)) + "\n\ntraining set size")
        print(len(trainingSetFeatures))
        print(len(trainingSetLabels))

    return validationSetLabels, validationSetFeatures, trainingSetLabels, trainingSetFeatures