import cv2 

# Load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('C:\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# the photo where the programm will detect the face
# img = cv2.imread('olimpia_stage.jpg')

webcam = cv2.VideoCapture(0)

while True:
    #read current frame
    successful_fram_read, frame = webcam.read()

    #Converts image to grayscale (so that it ist a headache to make the computer detect the face)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face cordinates
    face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangle around the face(s)
    for (x,y,z,h) in face_cordinates:
        cv2.rectangle(frame, (x,y), (x+z, y+h), (0, 255, 0), 2)

    cv2.imshow('face detoctor', frame)
    #pauses the programm so that the image does not imediatley disappear
    key = cv2.waitKey(1)

    #if you press q key then quit the programm
    if key==81 or key==113:
        break

    webcam.release()

print("Code Completed")