# functions for analyzing data accuracy

# TODO redefine positive and negative to use non-boolean data

# helper to generate data all scores use. reduces computation time
def genPerformanceScores(predicted, actual):
    # measure data for variables used for calculations
    falsePositives = 0
    truePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    for i in range(len(predicted)):
        # true
        if (predicted[i] == actual[i]):
            # true positive
            if (predicted[i] == 1):
                truePositives += 1
            # true negative
            else:
                trueNegatives += 1
        # false positive
        elif (predicted[i] == 1):
            falsePositives += 1
        # false negative
        elif (predicted[i] == 0):
            falseNegatives += 1
    return falsePositives, truePositives, trueNegatives, falseNegatives

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def accuracy(predicted, actual, givenScores = None):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual)
    # use scores generated from another function
    else:
        f_p = givenScores[0]
        t_p = givenScores[1]
        t_n = givenScores[2]
        f_n = givenScores[3]

    # accuracy
    return (t_p + t_n) / len(actual) * 100

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def precision(predicted, actual, givenScores = None):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual)
    # use scores generated from another function
    else:
        f_p = givenScores[0]
        t_p = givenScores[1]

    # precision
    precision = t_p / (t_p + f_p)
    return precision * 100

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def recall(predicted, actual, givenScores = None):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual)
    # use scores generated from another function
    else:
        t_p = givenScores[1]
        f_n = givenScores[3]

    # recall
    recall = t_p / (t_p + f_n)
    return recall * 100

# givenScores is optional pregenerated information from another call to a score function. reduces calculation time significantly.
def F1Score(predicted, actual, givenScores = None):
    # generate own scores
    if (givenScores == None):
        f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual)
        givenScores = (f_p, t_p, t_n, f_n)
    
    precisionVal = precision(predicted, actual, givenScores)
    recallVal = recall(predicted, actual, givenScores)

    # F1 score
    return (2 * precisionVal * recallVal) / (precisionVal + recallVal)

def printAllScores(predicted, actual):
    f_p, t_p, t_n, f_n = genPerformanceScores(predicted, actual)
    givenScores = (f_p, t_p, t_n, f_n)
    
    print("Combined Score Tests")
    print("\tAccuracy: " + str(accuracy(predicted,actual,givenScores)))
    print("\tPrecision: " + str(precision(predicted,actual,givenScores)))
    print("\tRecall: " + str(recall(predicted,actual,givenScores)))
    print("\tF1 Score: " + str(F1Score(predicted,actual,givenScores)))

def testScores():
    arr1 = [0,1,0,0,0,0,0,0]
    arr2 = [1,1,1,1,0,0,0,0]

    # test score functions independantly
    print("Individual Score Tests")
    print("\tAccuracy: " + str(accuracy(arr1,arr2)))
    print("\tPrecision: " + str(precision(arr1,arr2)))
    print("\tRecall: " + str(recall(arr1,arr2)))
    print("\tF1 Score: " + str(F1Score(arr1,arr2)))

    # test score functions together
    print()
    printAllScores(arr1, arr2)

# uncomment to test
# testScores()