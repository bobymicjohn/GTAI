import numpy as np
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import cv2
import time

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

def record_screen():
    while(True):
        # Grabs an 800 x 600 Window in upper left corner of the screen.
        printscreen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        
        cv2.imshow('Window', cascadeUS(cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
def main():
    record_screen()
        
if __name__ == "__main__":
    main()