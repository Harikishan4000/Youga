from flask import Flask, render_template, Response, jsonify
import cv2
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import base64


import functions as my_fc   ## User defined functions


app=Flask(__name__)


mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

pose_vid = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.5)

camera=cv2.VideoCapture("http://192.168.1.4:8080/video")
# camera=cv2.VideoCapture("0")
option=1
#youtube - Stream live Video from Mobile phone Camera with Python and OPen CV
#PlayStore - IP webcam

oc=0
def generate_frames():
    counter=0
    label=""
    problems=[]
    poseName=""

    while True:
        counter+=1
        landmarks=[]
        success, frame=camera.read()

        if not success:
            print("Unable to access camera")
            continue
        else:
            
            frame=cv2. flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            frame=cv2.resize(frame, (int(frame_width*(640/frame_height)), 640))
            if counter==30:
                frame, landmarks=my_fc.detectPose(frame, pose_vid, display=False)

            if landmarks and counter==30:
                frame, label, problems, poseName, flag=my_fc.classifyPose(landmarks, frame, option, display=False)
                counter=0    
            ret, buffer=cv2.imencode(".jpg", frame)
            frame=buffer.tobytes()
            
            # cv2.im
            # show("Youga feed", frame)
                    
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

        
        # print("reached")

        html_response = (
                '<img src="data:image/jpeg;base64,' + base64.b64encode(frame).decode() + '">' +
                '<div style="color: white">' + label + '</div>'
            )

        return html_response

        


        

    camera.release()
    # cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)
