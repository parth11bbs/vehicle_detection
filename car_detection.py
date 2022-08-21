import cv2

# Capture frames from a video
cap = cv2.VideoCapture("C:/Users/USER/PycharmProjects/Resources/traffic_video1.mp4")
# Note: There is no need to mention DataType in python, it finds it's DataType automatically

# Trained XML classifier describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier("C:/Users/USER/Desktop/computer vision/haarcascade_car.xml")

while True:     # That means, loop keeps running until the frame stops moving
    # Read frames from a video
    ret, frames = cap.read()

    frames = cv2.resize(frames, (640, 480))

    # frames = cv2.GaussianBlur(frames, (15, 15), 3)

    # Convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    # Que: Why are we converting it into gray scale image ?
    # Ans: Because processing gray scale image (consisting of only two values-0 & 1) is easier for computers.


    # Detects car of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)   # car_cascade stores trained xml file

    # print(cars)

    # To draw a rectangle in each bars
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display frames in a window
    cv2.imshow("Car detection", frames)

    # Wait for Esc key to stop
    if cv2.waitKey(100) == 27:   # 27 is a 'ASCII' code for escape character
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()

    # Note: Low quality video is preferred for processing, hence if video quality is high (4k or 1080p),
    # first convert it into lower quality.
