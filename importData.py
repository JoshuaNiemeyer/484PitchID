# imports the baseball training data

import numpy as np
import KNN
import testData.testDataManagement as tdm
import statisticScores

# imports the pitch data into a np array of float and integer data.
def importPitchData(filepath, doDebugPrint = False):
    data = np.loadtxt(filepath, delimiter=',')
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    if doDebugPrint:
        print('array of labels: shape ' + str(np.shape(labels)))
        print('array of feature matrix: shape ' + str(np.shape(features)))
        print('array of labels that exist: ' + str(np.array(np.unique(labels))))

    return labels, features

# test function
def testImport():
    labels, features = importPitchData("pitch_data_raw.csv", True)
    print("\ninteger IDs")
    print(labels[0])
    print(labels[500])
    print("\nfloat features")
    print(features[0][0])
    print(features[500][0])

    # split into validation and training set
    valLabels, valFeatures, trainLabels, trainFeatures = tdm.splitValidationTraining(labels, features, 20)

    print("\nValidation Set Size: " + str(len(valLabels)))
    print("Training Set Size: " + str(len(trainLabels)))

    # # test only on subset of data
    # valLabels = valLabels[0:1000]
    # valFeatures = valFeatures[0:1000]
    # trainLabels = trainLabels[0:1000]
    # trainFeatures = trainFeatures[0:1000]

    print("\nTesting data in Euclidean KNN")
    predictedLabels = KNN.kNearestNeighborsEuclidean(valFeatures, trainLabels, trainFeatures, 10)
    # print out all scores
    statisticScores.printAllScores(predictedLabels, valLabels)

    # test other KNN types
    print("\nTesting data in Manhatten KNN")
    predictedLabels = KNN.kNearestNeighborsManhatten(valFeatures, trainLabels, trainFeatures, 10)
    # print out all scores
    statisticScores.printAllScores(predictedLabels, valLabels)
    print("\nTesting data in Hamming KNN")
    predictedLabels = KNN.kNearestNeighborsHamming(valFeatures, trainLabels, trainFeatures, 10)
    # print out all scores
    statisticScores.printAllScores(predictedLabels, valLabels)

# uncomment to test
# testImport()