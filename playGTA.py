import numpy as np
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import cv2
import time
from directkeys import PressKey, W, A, S, D 

cascPath = 'Cascades/merge_cascade_updated.xml'
cascPath2 = 'Cascades/added_lane_cascade_updated.xml'
cascPath3 = 'Cascades/pedestrianCrossing_cascade.xml'
cascPath4 = 'Cascades/laneEnds_cascade.xml'
cascPath5 = 'Cascades/stop_cascade.xml'
cascPath6 = ''
cascPath7 = 'Cascades/signal_ahead_cascade.xml'
faceCasc = 'haarcascade_frontalface_default.xml'
mergeCascade = cv2.CascadeClassifier(cascPath)
addedLaneCascade = cv2.CascadeClassifier(cascPath2)
pedestrianCascade = cv2.CascadeClassifier(cascPath3)
laneEndsCascade = cv2.CascadeClassifier(cascPath4)
stopCascade = cv2.CascadeClassifier(cascPath5)
signalAheadCascade = cv2.CascadeClassifier(cascPath7)
faceCascade = cv2.CascadeClassifier(faceCasc)

def cascadeUS(frame):
	gray = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	merge = mergeCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(20, 20),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
    
	pedestrians = pedestrianCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
    
	stop = stopCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
	 

	for (x, y, w, h) in merge:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		
	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		
	for (x, y, w, h) in stop:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
		
	return frame

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

def pre_process(image):
    original_image = image
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    #ROI restriction
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    processed_img = roi(processed_img, [vertices])
    
    return processed_img

def main():
    while(True):
        #PressKey(W)
        # Grabs an 800 x 600 Window in upper left corner of the screen.
        screen_grab = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        processed_frame = pre_process(screen_grab)
        cv2.imshow('window', processed_frame)
        ##cv2.imshow('Window', cascadeUS(cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    main()