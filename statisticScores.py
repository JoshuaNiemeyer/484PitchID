# functions for analyzing data accuracy

PLAYER_IDS = [434378, 453286, 501985, 543037, 543243, 544931, 594798] # calc previously on import in importData.py

# helper to generate data all scores use. reduces computation time
# IDsList is the list of every the ID to check for the positive and negative identification of.
# run for every ID and sum to get total score for all scores
def genPerformanceScores(predicted, actual, IDsList = PLAYER_IDS):
    # measure data for variables used for calculations
    falsePositives = 0
    truePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    
    # run for every ID
    for IDnum in range(len(IDsList)):
        for i in range(len(predicted)):
            # true
            if (predicted[i] == actual[i]):
                # true positive
                if (predicted[i] == IDsList[IDnum]):
                    truePositives += 1
                # true negative
                else:
                    trueNegatives += 1
            # false positive
            elif (predicted[i] == IDsList[IDnum]):
                falsePositives += 1
            # false negative
            else:
                falseNegatives += 1

    # return a sumation of each type for every ID type
    return falsePositives, truePositives, trueNegatives, falseNegatives

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def accuracy(predicted, actual, givenScores = None, IDsList = PLAYER_IDS):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual, IDsList)
    # use scores generated from another function
    else:
        f_p = givenScores[0]
        t_p = givenScores[1]
        t_n = givenScores[2]
        f_n = givenScores[3]

    # accuracy
    return (t_p + t_n) / (t_p + t_n + f_p + f_n) * 100

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def precision(predicted, actual, givenScores = None, IDsList = PLAYER_IDS):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual, IDsList)
    # use scores generated from another function
    else:
        f_p = givenScores[0]
        t_p = givenScores[1]

    # precision
    precision = t_p / (t_p + f_p)
    return precision * 100

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def recall(predicted, actual, givenScores = None, IDsList = PLAYER_IDS):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual, IDsList)
    # use scores generated from another function
    else:
        t_p = givenScores[1]
        f_n = givenScores[3]

    # recall
    recall = t_p / (t_p + f_n)
    return recall * 100

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def F1Score(predicted, actual, givenScores = None, IDsList = PLAYER_IDS):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual, IDsList)
        givenScores = (f_p, t_p, t_n, f_n)
    
    precisionVal = precision(predicted, actual, givenScores)
    recallVal = recall(predicted, actual, givenScores)

    # F1 score
    return (2 * precisionVal * recallVal) / (precisionVal + recallVal)

def printAllScores(predicted, actual, IDsList = PLAYER_IDS):
    f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual, IDsList)
    givenScores = (f_p, t_p, t_n, f_n)
    
    print("Combined Score Tests")
    print("\tAccuracy: " + str(accuracy(predicted,actual,givenScores,IDsList)))
    print("\tPrecision: " + str(precision(predicted,actual,givenScores,IDsList)))
    print("\tRecall: " + str(recall(predicted,actual,givenScores,IDsList)))
    print("\tF1 Score: " + str(F1Score(predicted,actual,givenScores,IDsList)))

def testScores():
    arr1 = [0,1,0,0,0,0,0,0]
    arr2 = [1,1,1,1,0,0,0,0]
    IDs = [0,1]

    # test score functions independantly
    print("Individual Score Tests")
    print("\tAccuracy: " + str(accuracy(arr1,arr2,IDsList = IDs)))
    print("\tPrecision: " + str(precision(arr1,arr2,IDsList = IDs)))
    print("\tRecall: " + str(recall(arr1,arr2,IDsList = IDs)))
    print("\tF1 Score: " + str(F1Score(arr1,arr2,IDsList = IDs)))

    # test score functions together
    print()
    printAllScores(arr1, arr2,IDs)

# uncomment to test
# testScores()