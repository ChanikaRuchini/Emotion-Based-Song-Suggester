import numpy as np
import cv2
#Get real time face image
def getImage():
        cap = cv2.VideoCapture(0)
        count = 0

        while(count!=1):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imwrite('images/frame.png',frame)
            count+=1
            '''
            if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            '''
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()