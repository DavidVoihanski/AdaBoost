import numpy as np
import math


def trainCirc():
    """""
    Main function for training with circle.
    For every r from 1 to 8 run for 100 iterations (it) and randomly separate to 65 training 
    65 testing points, then  ind the r best circles, updating the weights every time based on the 
    AdaBoost algorithm. Every it, after training, we test and save the error in a list.
    After 100 iterations we check the average loss in the train and test with r circles.
    """""
    points = readFile("HC_Body_Temperature")
    for r in range(1, 9):
        errorListTrain = []
        errorListTest = []
        for it in range(0, 100):
            trainIndices = np.random.choice(points.shape[0], 65, replace=False)
            testIndices = np.zeros_like(trainIndices)
            index = 0
            for i in range(0, 130):
                if i not in trainIndices:
                    testIndices[index] = i
                    index += 1
            circList = adaBoostCirc(points, trainIndices, r)
            guessesTrain = np.ndarray(shape=[65])
            guessesTest = np.ndarray(shape=[65])
            for i in range(0, 65):
                point = points[trainIndices[i]]
                Fx = calcFxCirc(circList, point)
                if Fx >= 0:
                    guessesTrain[i] = 1
                else:
                    guessesTrain[i] = -1
                point = points[testIndices[i]]
                Fx = calcFxCirc(circList, point)
                if Fx >= 0:
                    guessesTest[i] = 1
                else:
                    guessesTest[i] = -1
            errorTest = 0
            errorTrain = 0
            for i in range(0, 65):
                if points[trainIndices[i], 2] != guessesTrain[i]:
                    errorTrain += 1
                if points[testIndices[i], 2] != guessesTest[i]:
                    errorTest += 1
            errorListTrain.append(errorTrain / 65)
            errorListTest.append(errorTest / 65)
        print("Average train error in round {} was".format(r), np.average(errorListTrain))
        print("Average test error in round {} was".format(r), np.average(errorListTest))


def trainRec():
    """""
    Main function for training with rectangles.
    For every r from 1 to 8 run for 100 iterations (it) and randomly separate to 65 training 
    65 testing points, then  ind the r best rectangles, updating the weights every time based on the 
    AdaBoost algorithm. Every it, after training, we test and save the error in a list.
    After 100 iterations we check the average loss in the train and test with r circles.
    """""
    points = readFile("HC_Body_Temperature")
    for r in range(1, 9):
        errorListTrain = []
        errorListTest = []
        for it in range(0, 100):
            trainIndices = np.random.choice(points.shape[0], 65, replace=False)
            testIndices = np.zeros_like(trainIndices)
            index = 0
            for i in range(0, 130):
                if i not in trainIndices:
                    testIndices[index] = i
                    index += 1
            recList = adaBoostRect(points, trainIndices, r)
            guessesTrain = np.ndarray(shape=[65])
            guessesTest = np.ndarray(shape=[65])
            for i in range(0, 65):
                point = points[trainIndices[i]]
                Fx = calcFxRect(recList, point)
                if Fx >= 0:
                    guessesTrain[i] = 1
                else:
                    guessesTrain[i] = -1
                point = points[testIndices[i]]
                Fx = calcFxRect(recList, point)
                if Fx >= 0:
                    guessesTest[i] = 1
                else:
                    guessesTest[i] = -1
            errorTest = 0
            errorTrain = 0
            for i in range(0, 65):
                if points[trainIndices[i], 2] != guessesTrain[i]:
                    errorTrain += 1
                if points[testIndices[i], 2] != guessesTest[i]:
                    errorTest += 1
            errorListTrain.append(errorTrain/65)
            errorListTest.append(errorTest/65)
        print("Average train error in round {} was".format(r), np.average(errorListTrain))
        print("Average test error in round {} was".format(r), np.average(errorListTest))


def calcFxCirc(circList: np.ndarray, point: np.ndarray) -> int:
    """
    Function for calculation the Fx for a given point.
    For every circle, if the point was classified as negative by that circle, the circle's
    alpha is subtracted from Fx and if it was classified as positive its alpha is added to Fx.
    After checking with every circle, if Fx is positive AdaBoost will classify the point as a positive,
    and as a negative otherwise.
    :param circList: The list of r best circles found beforehand
    :param point: The point we want to classify
    :return: The calculated Fx
    """
    Fx = 0
    for circ in circList:
        center = circ[0]
        edge = circ[1]
        alpha = circ[2]
        circType = circ[3]
        if isWithinCircle(center, edge, point):
            if circType == 1:
                Fx += alpha
            else:
                Fx -= alpha
        else: # outside circle
            if circType == 1:
                Fx -= alpha
            else:
                Fx += alpha
    return Fx


def calcFxRect(recList: np.ndarray, point: np.ndarray) -> int:
    """
    Function for calculation the Fx for a given point.
    For every rectangle, if the point was classified as negative by that rectangle, the rectangle's
    alpha is subtracted from Fx and if it was classified as positive its alpha is added to Fx.
    After checking with every rectangle, if Fx is positive AdaBoost will classify the point as a positive,
    and as a negative otherwise.
    :param recList: The list of r best rectangles found beforehand
    :param point: The point we want to classify
    :return: The calculated Fx
    """
    Fx = 0
    for rect in recList:
        first = rect[0]
        second = rect[1]
        alpha = rect[2]
        rectType = rect[3]
        if isWithinRec(first, second, point):
            if rectType == 1:
                Fx += alpha
            else:
                Fx -= alpha
        else:  # not in rec
            if rectType == 1:
                Fx -= alpha
            else:
                Fx += alpha
    return Fx


def isWithinRec(first: np.ndarray, second: np.ndarray, point: np.ndarray) -> bool:
    """
    Checks if a a point is in the rectangle created by first and second.
    :param first: The first point defining the rectangle
    :param second: The second point defining the rectangle
    :param point: The point we want to check
    :return: Returns true if the point is within the rectangle and false otherwise.
    """
    if first[0] > second[0]:
        temp = first
        first = second
        second = temp
    x1 = first[0]
    x2 = second[0]
    y1 = first[1]
    y2 = second[1]
    x = point[0]
    y = point[1]
    if x1 <= x <= x2:
        if y1 <= y2:
            if y1 <= y <= y2:
                return True
            else:
                return False
        else:  # y1 > y2
            if y2 <= y <= y1:
                return True
            else:
                return False
    else:
        return False


def circle(points: np.ndarray, indices: np.ndarray) -> (tuple, float, list, int):
    """
    Checks for the circle with the lowest error defined by 2 points, a center point and a point defining
    the edge of the circle. This means we need to check for (65 choose 2)*2 circles, once it is
    classified as + inside the circle then as - inside. Then we switch the center point with the edge point, this
    means we need to check for (65 choose 2)*2 + (65 choose 2)*2 circles in total.
    :param points: Numpy array of 130 points
    :param indices: The indices of points randomly chosen for training
    :return: tuple[center, edge] best circle found, its error, list of guesses defined by this circle, + or - inside
    """
    minError = 1
    bestPoints = (0, 0)
    bestGuess = []
    for i in range(0, 64):
        j = i+1
        while j < 65:
            firstIndex = indices[i]
            secondIndex = indices[j]
            first = points[firstIndex]
            second = points[secondIndex]
            guesses = np.ndarray(shape=[65])
            for k in range(0, 65):
                pointIndex = indices[k]
                point = points[pointIndex]
                if isWithinCircle(first, second, point):
                    guesses[k] = 1
                else:
                    guesses[k] = -1
            error = 0
            guessedInside = 1
            for k in range(0, 65):
                point = points[indices[k]]
                if guesses[k] != point[2]:
                    error += point[3]
            if error < minError:
                minError = error
                bestGuess = guesses
                bestPoints = (points[indices[i]], points[indices[j]])
            # guess "-" inside
            for k in range(0, 65):
                pointIndex = indices[k]
                point = points[pointIndex]
                if isWithinCircle(first, second, point):
                    guesses[k] = -1
                else:
                    guesses[k] = 1
            error = 0
            for k in range(0, 65):
                point = points[indices[k]]
                if guesses[k] != point[2]:
                    error += point[3]
            if error < minError:
                bestPoints = (points[indices[i]], points[indices[j]])
                minError = error
                bestGuess = guesses
                guessedInside = -1

            # swap center and edge
            for k in range(0, 65):
                pointIndex = indices[k]
                point = points[pointIndex]
                if isWithinCircle(second, first, point):
                    guesses[k] = 1
                else:
                    guesses[k] = -1
            error = 0
            for k in range(0, 65):
                point = points[indices[k]]
                if guesses[k] != point[2]:
                    error += point[3]
            if error < minError:
                minError = error
                bestGuess = guesses
                guessedInside = 1
                bestPoints = (points[indices[j]], points[indices[i]])
            # guess "-" inside
            for k in range(0, 65):
                pointIndex = indices[k]
                point = points[pointIndex]
                if isWithinCircle(second, first, point):
                    guesses[k] = -1
                else:
                    guesses[k] = 1
            error = 0
            for k in range(0, 65):
                point = points[indices[k]]
                if guesses[k] != point[2]:
                    error += point[3]
            if error < minError:
                bestPoints = (points[indices[j]], points[indices[i]])
                minError = error
                bestGuess = guesses
                guessedInside = -1

            j += 1
    return bestPoints, minError, bestGuess, guessedInside


def isWithinCircle(center, edge, point) -> bool:
    """
    Checks if a given point is within a circle defined by a center point and an edge point
    :param center: The point defining the center
    :param edge: The point defining the edge
    :param point: The point we want to check
    :return: Returns true if the point is inside the circle and false otherwise
    """
    x = center[0]
    y = center[1]
    rX = edge[0]
    rY = edge[1]
    disX = math.pow(x - rX, 2)
    disY = math.pow(y - rY, 2)
    xP = point[0]
    yP = point[1]
    disXP = math.pow(x - xP, 2)
    disYP = math.pow(y - yP, 2)
    if disXP + disYP <= disX + disY:
        return True
    else:
        return False


def readFile(pathToFile: str) -> np.ndarray:
    """
    Reads the file and converts each person to a (x, y) point defined by body weight and temperature with its label
    (1 male and -1 female).
    Defines every point's weight as 1/65 since it's the training set size.
    :param pathToFile:
    :return: List of points parsed from the file
    """
    file = open(pathToFile, "r")
    points = np.zeros(shape=[130, 4])
    index = 0
    for line in file:
        temp = line.split()
        temp = np.array(temp).astype(float)
        points[index, 0] = temp[0]
        points[index, 1] = temp[2]
        if temp[1] == 1:
            points[index, 2] = 1
        else:
            points[index, 2] = -1
        points[index, 3] = 1/65
        index += 1
    return points


def rectangle(points: np.ndarray, indices: np.ndarray) -> (tuple, float, list, int):
    """
    Checks for the rectangle with the lowest error defined by 2 points, a left corner point and a right corner point.
    This means we need to check for (65 choose 2)*2 rectangles, once it is
    classified as + inside the rectangle then as - inside.
    :param points: Numpy array of 130 points
    :param indices: The indices of points randomly chosen for training
    :return: tuple[leftPoint, rightPoint] the rec found, its error, list of guesses defined by this rec, + or - inside
    """
    minError = 1
    bestPoints = (0, 0)
    bestGuess = []
    for i in range(0, 64):
        j = i+1
        while j < 65:
            first = points[indices[i]]
            second = points[indices[j]]
            guesses = np.ndarray(shape=[65])
            for k in range(0, 65):
                pointIndex = indices[k]
                point = points[pointIndex]
                if isWithinRec(first, second, point):
                    guesses[k] = 1
                else:
                    guesses[k] = -1
            # guess + inside
            error = 0
            guessedInside = 1
            for k in range(0, 65):
                point = points[indices[k]]
                if guesses[k] != point[2]:
                    error += point[3]
            if error < minError:
                bestPoints = (points[indices[i]], points[indices[j]])
                minError = error
                bestGuess = guesses
            for k in range(0, 65):
                pointIndex = indices[k]
                point = points[pointIndex]
                if isWithinRec(first, second, point):
                    guesses[k] = -1
                else:
                    guesses[k] = 1
            error = 0
            for k in range(0, 65):
                point = points[indices[k]]
                if guesses[k] != point[2]:
                    error += point[3]
            if error < minError:
                bestPoints = (points[indices[i]], points[indices[j]])
                minError = error
                bestGuess = guesses
                guessedInside = -1

            j += 1
    return bestPoints, minError, bestGuess, guessedInside


def adaBoostRect(points: np.ndarray, indices: np.ndarray, it: int) -> list:
    """
    For every i from 0 to it, find a the best rectangle and update the weights of the
    training points based on the following schema:
        Calculate the rectangle's alpha defined as 0.5*log((1- error)/error)
        If got an error on point p, increase its weight by multiplying it by exp(alpha) < 1
        If guessed point p correctly, decrease its weight by multiplying it by exp(-alpha) > 1
    :param points: Numpy array of 130 points
    :param indices: 65 randomly chosen training points
    :param it: number of rectangles we need to find
    :return: Returns a list of the best rectangles found, with their corresponding alpha and type (+ or - inside)
    """
    recList = []
    for i in range(0, 65):
        points[indices[i], 3] = 1/65
    for i in range(0, it):
        rec, error, guesses, recType = rectangle(points, indices)
        alpha = 0.5 * math.log((1 - error) / error)
        tempRec = [rec[0], rec[1], alpha, recType]
        recList.append(tempRec)
        totalW = 0
        for j in range(0, 65):
            if guesses[j] != points[indices[j], 2]:
                points[indices[j], 3] = points[indices[j], 3] * math.exp(alpha)
            else:
                points[indices[j], 3] = points[indices[j], 3] * math.exp(-alpha)
            totalW += points[indices[j], 3]

        for j in range(0, 65):
            points[indices[j], 3] = points[indices[j], 3]/totalW
    return recList


def adaBoostCirc(points: np.ndarray, indices: np.ndarray, it: int) -> list:
    """
    For every i from 0 to it, find a the best circle and update the weights of the
    training points based on the following schema:
        Calculate the circle's alpha defined as 0.5*log((1- error)/error)
        If got an error on point p, increase its weight by multiplying it by exp(alpha) < 1
        If guessed point p correctly, decrease its weight by multiplying it by exp(-alpha) > 1
    :param points: Numpy array of 130 points
    :param indices: 65 randomly chosen training points
    :param it: number of circles we need to find
    :return: Returns a list of the best circle found, with their corresponding alpha and type (+ or - inside)
    """
    circList = []
    for i in range(0, 65):
        points[indices[i], 3] = 1/65
    for i in range(0, it):
        circ, error, guesses, circType = circle(points, indices)
        alpha = 0.5 * math.log((1 - error) / error)
        tempCirc = [circ[0], circ[1], alpha, circType]
        circList.append(tempCirc)
        totalW = 0
        for j in range(0, 65):
            if guesses[j] != points[indices[j], 2]:
                points[indices[j], 3] = points[indices[j], 3]*math.exp(alpha)
            else:
                points[indices[j], 3] = points[indices[j], 3]*math.exp(-alpha)
            totalW += points[indices[j], 3]
        for j in range(0, 65):
            points[indices[j], 3] = points[indices[j], 3]/totalW
        return circList


if __name__ == "__main__":
    print("rectangle")
    # trainRec()
    print("circle")
    trainCirc()
