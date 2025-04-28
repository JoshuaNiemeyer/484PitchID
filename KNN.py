import numpy as np
import statistics

import testData.testDataManagement as tdm
import statisticScores as statTest

# K Nearest neighbor algorithms. uncomment at bottom of file to test
def testKNN():
    # load test data
    testLabels, testFeatures = tdm.loadTestDataHW("testData/trainingDataHW1.txt")
    # split into validation and training set
    valLabels, valFeatures, trainLabels, trainFeatures = tdm.splitValidationTraining(testLabels, testFeatures, 20)
    
    # predict with KNN euclidean
    print("\nRUNNING EUCLIDEAN KNN ON TEST DATA")
    predictedLabels = kNearestNeighborsEuclidean(valFeatures, trainLabels, trainFeatures, 10)
    # print out all scores
    statTest.printAllScores(predictedLabels, valLabels)
    
    # predict with KNN manhatten
    print("\nRUNNING MANHATTEN KNN ON TEST DATA")
    predictedLabels = kNearestNeighborsManhatten(valFeatures, trainLabels, trainFeatures, 10)
    # print out all scores
    statTest.printAllScores(predictedLabels, valLabels)

    # predict with KNN hamming
    print("\nRUNNING HAMMING KNN ON TEST DATA")
    predictedLabels = kNearestNeighborsHamming(valFeatures, trainLabels, trainFeatures, 10)
    # print out all scores
    statTest.printAllScores(predictedLabels, valLabels)

# generic KNN requiring distance selection to work
# K is number of adjacent data points to match to
# distance Method is type of distance to use. currently supports "Euclidean", "Hamming", and "Manhatten". defaults to "Euclidean"
def kNearestNeighbors(features, trainingLabels, trainingFeatures, K, distanceMethod = "Euclidean"):
    # change to np arrays
    trainingFeatures = np.array(trainingFeatures)
    features = np.array(features)
    
    # predict a label for every feature
    PredictedLabels = []
    for i in range(len(features)):
        # get distance to every feature in training set
        distancesToFeature = []
        for j in range(len(trainingFeatures)):
            if (distanceMethod == "Manhatten"):
                # manhatten distance between two features
                distancesToFeature.append(np.sum(np.absolute(features[i] - trainingFeatures[j])))
            elif (distanceMethod == "Hamming"):
                # hamming distance between two features
                distancesToFeature.append(np.sum(int(not(np.array_equal(features[i], trainingFeatures[j])))))
            else:
                # euclidean distance between two features
                distancesToFeature.append(np.sum(np.square(features[i] - trainingFeatures[j])))
        
        # get labels from K smallest euclidean distances
        closestLabelsToFeature = []
        trainingLabelsCopy = trainingLabels[:] # copy so can delete and still use list for other iterations
        for j in range(K):
            minDistanceIndex = np.argmin(distancesToFeature)
            closestLabelsToFeature.append(trainingLabelsCopy[minDistanceIndex])
            # remove minimum item from lists
            distancesToFeature.pop(minDistanceIndex)
            trainingLabelsCopy.pop(minDistanceIndex)
        PredictedLabels.append(statistics.mode(closestLabelsToFeature)) # add most often label as this feature's prediction        
    return PredictedLabels

# shorthand methods
def kNearestNeighborsEuclidean(features, trainingLabels, trainingFeatures, K):
    return kNearestNeighbors(features, trainingLabels, trainingFeatures, K)

def kNearestNeighborsManhatten(features, trainingLabels, trainingFeatures, K):
    return kNearestNeighbors(features, trainingLabels, trainingFeatures, K, "Manhatten")

def kNearestNeighborsHamming(features, trainingLabels, trainingFeatures, K):
    return kNearestNeighbors(features, trainingLabels, trainingFeatures, K, "Hamming")

# uncomment to test algorithm
# testKNN()