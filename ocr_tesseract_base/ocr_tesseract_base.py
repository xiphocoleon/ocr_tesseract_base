from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2 as cv
from matplotlib import pyplot as plt

def decode_predictions(scores, geometry):
    #get amount of rows, cols, from score volume
    #initialize our bounding boxes and confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    #loop over num of rows
    for y in range(0, numRows):
        #extract scores(probabilities), followed by geometrical data
        #used to derive potential bounding boxes coords around text
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        #loop over number of cols
        for x in range(0, numCols):
            #if score prob below thresh, ignore
            if scoresData[x] < args["min_confidence"]:
                continue

            #compute offset as our resulting feature
            #maps will be 4x smaller than input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            #extract the rotation angle for the prediction and 
            #then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            #use the geometry volume to derive the width and height
            #of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            #compute both starting and ending (x, y)-coordinates
            #for the text predicting box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            #add the bounding box coordinates and prob score to our rep lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    #return tuple of bounding boxes and associated confidences
    return (rects, confidences)

#construct argument parser and parse args

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

#load the input image and grab the image dimensions
image = cv.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]

#set the new widht and height and then detemine the raio in change
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

#resize the image and grab the new image dimensions
image = cv.resize(image, (newW, newH))
(H, W) = image.shape[:2]

#define output layer name for EAST detector model 
#the first is the output probabilities
#the second is the bounding box coords
layerNames = [
    "feature_fusion/Conv_7/Sigmoid", 
    "feature_fusion/concat_3"]

#load the pretrained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv.dnn.readNet(args["east"])

#construct a blob from the image and perform forward pass of
#the model to obtain the two output layer sets
blob = cv.dnn.blobFromImage(image, 1.0, (W, H), 
                            (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

#decode the predictions then apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

#initialize the results
results = []

#loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    #scale the bounding box coordinates based on the respective ratios
    startX = int(startX * rW)
    startY = int(startY *rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    #to get better OCR, apply padding around bounding box
    #find x and y deltas
    dX = int((endX - startX) * args["padding"])
    dY = int((endY - startY) * args["padding"])

    #apply padding
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    #extract actual padded ROI
    roi = orig[startY:endY, startX:endX]

    #for Tesseract apply a language, and OEM flag of 1 to use
    #LSTM neural net model for OCR, and an OEM value, 7, 
    #to treat ROI as single line of text
    config = ("-1 eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, lang='eng')

    #add the bounding box coords and ocr'd text to list of results
    results.append(((startX, startY, endX, endY), text))

#sort results bounding box from top to bottom
results = sorted(results, key=lambda r:r[0][1])

#loop over results
for ((startX, startY, endX, endY), text) in results:
    #display the text OCR'd
    print("OCR TEXT")
    print("=======")
    print("{}\n".format(text))

    #string out the non-ASCII text so we can draw text on the image
    #w/ OpenCV; draw the text and a bounding box surrounding the text
    #region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv.rectangle(output, (startX, startY), (endX, endY), 
                 (0, 0, 255), 2)
    cv.putText(output, text, (startX, startY - 20), 
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    #show output image
    #img = cv.imread('path_to_image')
    plt.imshow(output, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()




