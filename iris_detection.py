import cv2 as cv
import os
import numpy as np

eyesList = []
radius = 0
currentEye = 0
centroid = (0,0)

def getNewEye(list):
	global currentEye
	if (currentEye >= len(list)):
		currentEye = 0
	newEye = list[currentEye]
	currentEye += 1
	return (newEye)

def getIris(frame):
    iris = []
    copyImg = frame.copy()
    resImg = frame.copy()
    grayImg = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    cv.cvtColor(frame, cv.COLOR_BGR2GRAY, grayImg)
    cv.Canny(grayImg, 5, 70, grayImg, 3)
    grayImg = cv.GaussianBlur(grayImg,(7,7), 0, 0)
    circles = getCircles(grayImg)
    iris.append(resImg)
    
    for circle in circles:
        rad = int(circle[0][2])
        global radius
        radius = rad
        cv.circle(mask, centroid, rad, (255,255,255), cv.FILLED)
        cv.circle(copyImg, centroid, rad, (0,0,0), 1)
        cv.bitwise_not(mask, mask)
        cv.subtract(frame,copyImg, resImg, mask)
        
        x = int(centroid[0] - rad)
        y = int(centroid[1] - rad)
        w = int(rad * 2)
        h = w
        roi = resImg[y:y+h , x:x+w]

        cv.imshow('cropped iris', roi)
        return (copyImg)
    return (resImg)

def getCircles(image):
    i = 80
    while(i<151):
        circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 2, 100.0,param1 = 30, param2= i, minRadius = 60, maxRadius = 110)
        if(len(circles) == 1):
            return circles
        i += 1
    return ([])

def getPupil(frame):
    pupilImg = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    cv.inRange(frame, (10,10,10), (80,80,80), pupilImg)
    contours, hierarchy = cv.findContours(pupilImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    del pupilImg
    pupilImg = frame.copy()
    contourIdx = 0
    for c in contours:
        moments = cv.moments(c)
        area = moments['m00']
        if (area > 50):
            pupilArea = area
            x = moments['m10']/area
            y = moments['m01']/area
            pupil = contours
            global centroid
            centroid = (int(x), int(y))
            cv.drawContours(pupilImg, pupil, contourIdx ,color = (0,0,0), thickness =  cv.FILLED)
            break
        contourIdx += 1
    return (pupilImg)

cv.namedWindow('input', cv.WINDOW_AUTOSIZE)

eyesList = os.listdir('dataset/eyes')

key = 0
while True:
    eye = getNewEye(eyesList)
    frame = cv.imread('dataset/eyes/'+eye)
    iris = frame.copy()
    output = getPupil(frame)
    iris = getIris(output)
    cv.imshow('input', frame)
    cv.imshow('iris', iris)
    
    key = cv.waitKey(3000)
    if (key == 27 or key == 1048603):
	    break

cv.destroyAllWindows()
