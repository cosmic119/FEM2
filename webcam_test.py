import numpy as np
import cv2
import os
import tensorflow as tf
import facial_expression_with_landmark
import dlib

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
landmark_data = []

def get_emotion_by_index(index):
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    if index == 0:
        return "Angry"
    elif index == 1:
        return "Disgust"
    elif index == 2:
        return "Fear"
    elif index == 3:
        return "Happy"
    elif index == 4:
        return "Sad"
    elif index == 5:
        return "Surprise"
    elif index == 6:
        return "Neutral"
    else:
        return "Unregistered emotion"



def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(0, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)

        for i in range(0, 8):
            landmarks_vectorised.append(0)

        # self.data['landmarks_vectorised'] = landmarks_vectorised
        landmark_data.append(landmarks_vectorised)

    if len(detections) < 1:
        landmarks_vectorised_x = "error"
        landmarks_vectorised = []
        for i in range(0, 144):
            landmarks_vectorised.append(0)

        landmark_data.append(landmarks_vectorised)
        return False


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

sess = tf.Session()
face_expression_detector = facial_expression_with_landmark.facial_expression()
checkpoint_save_dir = os.path.join("/home/hci/PycharmProjects/hklovelovehs/hialove")
face_expression_detector.load_graph(sess, checkpoint_save_dir)

preferred_w, preferred_h = 800, 600

res = np.array([[0]])
sentiment_argmax = 0
sentiment_arr = []

while True:
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]

    frame = cv2.resize(frame, None, fx=preferred_w / frame_width, fy=preferred_h / frame_height,
                       interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width = frame.shape[:2]

    grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayed, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = grayed[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        desired_h, desired_w = float(48), float(48)
        resized_ratio_h, resized_ratio_w = desired_h / h, desired_w / w
        res = cv2.resize(roi_gray, None, fx=resized_ratio_w, fy=resized_ratio_h, interpolation=cv2.INTER_CUBIC)
        #get landmark
        get_landmarks(res)
        print(res)
        print(landmark_data)




        # face_expression_detector.get_landmark(res)
        # tf.concat([res, face_expression_detector.landmark_data],1)
        # facial_expression_with_landmark.gwrt

        res = np.reshape(res, (-1, 2304))
        landmark_res = np.reshape(landmark_data, (-1, 144))

        # feed_dict = {face_expression_detector.X:res,face_expression_detector.landmark_X : landmark_res, face_expression_detector.keep_prob:1}
        feed_dict = {face_expression_detector.X: res, face_expression_detector.landmark_X: landmark_res}
        sentiment_arr = np.array(sess.run(face_expression_detector.softmax_logits, feed_dict=feed_dict))
        landmark_data = []
        print(sentiment_arr)
        sentiment_arr = sentiment_arr[0]
        sentiment_argmax = np.argmax(sentiment_arr, axis=0)
        res = np.reshape(res, (48, 48))
        font_offset = 50
    for i, sentiment in enumerate(sentiment_arr):  # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        sentiment *= 100
        sentiment = round(sentiment, 3)
        if (sentiment_argmax == i):
            frame = cv2.putText(frame, get_emotion_by_index(i) + " " + str(sentiment),
                                (preferred_w - 300, i * font_offset + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = cv2.putText(frame, get_emotion_by_index(i) + " " + str(sentiment),
                                (preferred_w - 300, i * font_offset + 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("res", res)
    cv2.imshow("main frame", frame)
    if (cv2.waitKey(1) & 0xff == ord('q')):
        break
sess.close()
cap.release()
cv2.destroyAllWindows()
