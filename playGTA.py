import numpy as np
from PIL import ImageGrab
import cv2
import time

def record_screen():
    while(True):
        # Grabs an 800 x 600 Window in upper left corner of the screen.
        printscreen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        
        cv2.imshow('Window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
def main():
    record_screen()
        
if __name__ == "__main__":
    main()