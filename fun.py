from keras.models import load_model
import cv2
import numpy as np

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
gender_classifier = load_model('simple_CNN.81-0.96.hdf5', compile = False)
emotion_classifier = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5', compile = False)


gender_labels = ['woman', 'man']
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

et_size = emotion_classifier.input_shape[1:3]
gt_size = gender_classifier.input_shape[1:3]

font =  cv2.FONT_HERSHEY_SIMPLEX
var1 = cv2.COLOR_BGR2RGB
var2 = cv2.COLOR_BGR2GRAY


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def ConnectCam():
    """ find the indices of all cameras cannected to the camputer"""
    cams = []
    for i in range(200):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cams.append(i)

    return cams

def cam(frame, hei, wid):
    """ Detect faces, gender and emotions on a frame and then print somme statistics in the frames"""

    emo_count = [0, 0, 0, 0, 0, 0, 0]
    gen_count = [0, 0]


    rgb_image = cv2.cvtColor(frame, var1)
    gray_image = cv2.cvtColor(frame, var2)

    faces = face_detector.detectMultiScale(rgb_image, 1.3, 5)



    for face_coordinates in faces:
        x, y, w, h = face_coordinates

        rgb_face = rgb_image[(x-30):(x+w+30), (y-60):(y+h+60)]
        gray_face = gray_image[(x-20):(x+w+20), (y-40):(y+h+40)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        try:
            rgb_face = cv2.resize(rgb_face, (et_size))
            gray_face = cv2.resize(gray_face, (gt_size))
        except:
            continue

        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_arg = np.argmax(gender_classifier.predict(rgb_face))
        gen_count[gender_arg] += 1

        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        prob_emotions = emotion_classifier.predict(gray_face)

        emotion_arg = np.argmax(prob_emotions)

        emo_count[emotion_arg] += 1

        if gender_arg == 0:
            gen = "F "
        else :
            gen = "M "
        emo = str(emotion_labels[emotion_arg]) + ' '
        pro = str(prob_emotions[0][emotion_arg]*100)[0:5] + ' '
        cv2.putText(frame, (gen + emo + pro), (x,y-25), font, h/200, (0,0,255), 1, cv2.LINE_AA)




    cv2.putText(frame, 'Women : ' + str(gen_count[0]), (15,hei-15), font, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Men : '+ str(gen_count[1]), (15,hei-45), font, 1, (255,0,0), 1, cv2.LINE_AA)

    cv2.putText(frame, 'Angry : '+ str(emo_count[0]), (15,25), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Disgust : '+ str(emo_count[1]), (15,40), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Fear : '+ str(emo_count[2]), (15,55), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Happy : '+ str(emo_count[3]), (15,70), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Sad : '+ str(emo_count[4]), (15,85), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Surprise : '+ str(emo_count[5]), (15,100), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Neutral : '+ str(emo_count[6]), (15,115), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
