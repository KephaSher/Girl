import cv2
from fer import FER

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale) 
    dimensions = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

detector = FER(mtcnn=True)
camera = cv2.VideoCapture(0)

while True:
    _, img = camera.read()
    img = rescaleFrame(img)

    faces = detector.detect_emotions(img)
    for face in faces:
        emotion = detector.top_emotion(img)
        box = face['box']

        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv2.putText(img, emotion[0] + " " + str(round(emotion[1] * 100, 1)) + "%", 
            (x - 10, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), thickness=2)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
