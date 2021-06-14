import cv2 ,time

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier("smile.xml")

video=cv2.VideoCapture(0) #open webcam

while True:
    check,frame=video.read() #reading image coming out of camera, ret checks it and img saves it
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#to grey scale image as casccades trained on gray scale img
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        smile=smile_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)#loading cascade(properties) to read the image and load the images scalefactor stores size of image, minNeighbours define the min no of faces to be detected and minScale defines the size of face and bounding box after reading the face(coloured image)
        for x,y,w,h in smile:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3) #looping within the smile cascade to find whether there is atleast 1 smile or not
        # and as soon as there is one smile we put the same coordinates as the face and render the word smiling

    cv2.imshow('gotcha',frame)
    key=cv2.waitKey(1)#take video o/p

    if key==ord('q'):
         break# press 'esc to quit

video.release()
cv2.destroyAllWindows        
