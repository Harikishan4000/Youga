import cv2
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculateAngle( land1, land2, land3):
    #get landmark coordinates
    x1, y1, _ = land1
    x2, y2, _ = land2
    x3, y3, _ = land3

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle<0:
        angle+=360

    return angle
    
def reqAngles(landmarks, *args):
    retList=()
    for arg in args:
        if arg=='le':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
            
        elif arg=='re':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
             
        elif arg=='ls':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        elif arg=='rs':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        elif arg=='lk':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        elif arg=='rk':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        elif arg=='lh':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            
        elif arg=='rh':
            temp = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        retList+=(temp,)

    return retList
        
        
def correction(minmax, dict):
    probs=[]
    
    if "re" in dict:
        if dict['re']<minmax[0]:
            probs.append("Straighten right elbow a little more")
        elif dict['re']>minmax[1]:
            probs.append("Bend your right elbow more")
        

    if "le" in dict:
        if dict['le']<minmax[2]:
            probs.append("Straighten left elbow a little more")
        elif dict['le']>minmax[3]:
            probs.append("Bend your left elbow more")
        
        

    if "rh" in dict:
        if dict['rh']<minmax[4]:
            probs.append("Straighten your hip")
        elif dict['rh']>minmax[5]:
            probs.append("Bend your hip")
        
    if "lh" in dict:
        if dict['rh']<minmax[6]:
            probs.append("Straighten your hip")
        elif dict['rh']>minmax[7]:
            probs.append("Bend your hip")    
    
    if "rk" in dict:
        if dict['rk']<minmax[8]:
            probs.append("Straighten right your knee")
        elif dict['rk']>minmax[9]:
            probs.append("Bend your right knee")
        
        

    if "lk" in dict:
        if dict['lk']<minmax[10]:
            probs.append("Straighten your left knee")
        elif dict['lk']>minmax[11]:
            probs.append("Bend your left knee")
        
        
        
    if "rs" in dict:
        if dict['rs']<minmax[12]:
            probs.append("Raise your right hand higher")
        elif dict['rs']>minmax[13]:
            probs.append("Lower your right hand")
        
        

    if "ls" in dict:
        if dict['ls']<minmax[14]:
            probs.append("Raise your left hand higher")
        elif dict['ls']>minmax[15]:
            probs.append("Lower your left hand")
        
        
    
    return probs 


def tpose(landmarks):
    label='Unknown'
    re, le, rs, ls = reqAngles(landmarks, 're', 'le', 'rs', 'ls')
    
    # ret, let, rst, lst, rkt, lkt, rht, lht= -1, -1, -1, -1, -1, -1, -1, -1  ##Temp variables

    # remin, remax = 150, 200
    # rhmin, rhmax= 60, 120
    # rkmin, rkmax= 150, 200
    # rsmin, rsmax = 150, 200

    minmax= [165, 195, 160, 195, 0, 0, 0, 0, 0, 0, 0, 0, 80, 120, 80, 120]
    corrVal= {}
    probs=[]
    
        # T pose
    if re>minmax[0] and re<minmax[1]:
        pass
    else:
        corrVal['re']=re
    
    if le>minmax[2] and le<minmax[3]:
        pass
    else:
        corrVal['le']=le

    if rs>minmax[12] and rs<minmax[13]:
        pass
    else:
        corrVal['rs']=rs

    if ls>minmax[14] and ls<minmax[15]:
        pass
    else:
        corrVal['ls']=ls
        
    
    if corrVal:
        probs = correction(minmax, corrVal)
    elif not corrVal:
        label= "Tpose"
        probs=[]
    
    return label, probs

def namaskara(landmarks):
    label='Unknown'
    re, le, rk, lk, rs, ls = reqAngles(landmarks, 're', 'le','rk', 'lk', 'rs', 'ls')
    
    # ret, let, rst, lst, rkt, lkt, rht, lht= -1, -1, -1, -1, -1, -1, -1, -1  ##Temp variables

    # remin, remax = 150, 200
    # rhmin, rhmax= 60, 120
    # rkmin, rkmax= 150, 200
    # rsmin, rsmax = 150, 200

    minmax= [35, 75, 295, 335, 0, 0, 0, 0, 150, 200, 150, 200, 0, 40, 0, 40]
    corrVal= {}
    probs=[]
    
        # Namaskarasana
    if re>minmax[0] and re<minmax[1]:
        pass
    else:
        corrVal['re']=re
    
    if le>minmax[2] and le<minmax[3]:
        pass
    else:
        corrVal['le']=le

    if rk>minmax[8] and rk<minmax[9]:
        pass
    else:
        corrVal['rs']=rs

    if lk>minmax[10] and lk<minmax[11]:
        pass
    else:
        corrVal['ls']=ls    

    if rs>minmax[12] and rs<minmax[13]:
        pass
    else:
        corrVal['rs']=rs

    if ls>minmax[14] and ls<minmax[15]:
        pass
    else:
        corrVal['ls']=ls
        
    
    if corrVal:
        probs = correction(minmax, corrVal)
    elif not corrVal:
        label= "Namaskarasana"
        probs=[]
    
    return label, probs


def vriksh(landmarks):
    label='Unknown'
    
    rk, lk = reqAngles(landmarks,'rk', 'lk')


    minmax= [0, 0, 0, 0, 0, 0, 0, 0, 15, 90, 0, 0, 0, 0, 0, 0]
    corrVal= {}
    probs=[]

    if rk>minmax[8] and rk<minmax[9]:
        pass
    else:
        corrVal['rk']=rk

    if corrVal:
        probs = correction(minmax, corrVal)
    elif not corrVal:
        label= "RVrikshasana"
        probs=[]
    
    return label, probs
            
    
            
def veera(landmarks):
    label='Unknown'
    
    re, le, rs, ls, rk, lk, rh, lh = reqAngles(landmarks, 're', 'le', 'rs', 'ls', 'rk', 'lk', 'rh', 'lh')
    
    straightArms=False
    orientation='' #variable to keep track of orientation
    
    if re<195 and re >165  and le<195 and le>160:
        if ls>70 and ls<110 and rs<120 and rs>70:
            ## hands are straight
            straightArms=True
            
    if straightArms:
        ##Hip position check
        if rk>70 and rk<150:  #check if right knee is bent
            if lk>160 and lk<190:
                label="L - VeeraBhadrasana"
            
        elif lk>220 and lk<310:  #check if left knee is bent
            if rk>160 and rk<190:
                label="R - VeeraBhadrasana"
    return label

def vajrasana(landmarks):
    label='Unknown'
    
    rk, rh = reqAngles(landmarks, 'rk', 'rh')

    minmax= [0, 0, 0, 0, 85, 105, 0, 0, 300, 350, 0, 0, 0, 0, 0, 0]
    corrVal= {}
    probs=[]

    if rh>minmax[4] and rh<minmax[5]:
        pass
    else:
        corrVal['rh']=rh

    if rk>minmax[8] and rk<minmax[9]:
        pass
    else:
        corrVal['rk']=rk

    if corrVal:
        probs = correction(minmax, corrVal)
    elif not corrVal:
        label= "R Vajrasana"
        probs=[]
    
    return label, probs

    # if rh>80 and rh<120:  ##check if back is straight
    #     if rk>260 and rk<350:
    #         label ="Right facing Vajrasana"

    # if rh>240 and rh<290:  ##check if back is straight
    #     if rk>0 and rk<40:
    #         label ="Left facing Vajrasana"
    

def downdog(landmarks):
    label='Unknown'
    re, le, rs, ls, rk, lk, rh, lh = reqAngles(landmarks, 're', 'le', 'rs', 'ls', 'rk', 'lk', 'rh', 'lh')

    # ret, let, rst, lst, rkt, lkt, rht, lht= -1, -1, -1, -1, -1, -1, -1, -1  ##Temp variables

    # remin, remax = 150, 200
    # rhmin, rhmax= 60, 120
    # rkmin, rkmax= 150, 200
    # rsmin, rsmax = 150, 200

    minmax= [150, 200, 0, 0, 60, 120, 0, 0, 50, 200,0, 0, 150, 200, 0, 0]
    corrVal= {}
    probs=[]
    
    ### right facing
    if re>minmax[0] and re<minmax[1]: ##Right facing - elbow
        pass
    else:
        corrVal['re']=re

    if rh>minmax[5] and rh<minmax[6]: ##Right facing - hip
        pass
    else:
        corrVal['rh']=rh
    
    if rk>minmax[8] and rk<minmax[9]:
        pass
    else:
        corrVal['rk']=rk
    
    if rs>minmax[12] and rs<minmax[13]:
        pass
    else:
        corrVal['rs']=rs
    
    
    if corrVal:
        probs = correction(minmax, corrVal)
    elif not corrVal:
        label= "Downward dog"
        probs=[]
    
    return label, probs

def classifyPose(landmarks, output_image, option, display=False):
    label = 'Unknown'
    problems = ["Perfect! Youre great at this!"]
    labelColor = (0, 0, 255)
    problemColor = (0, 255, 255)

    '''
    re : right elbow
    le : left elbow
    
    rs : right shoulder
    ls : left shoulder

    rk : right knee
    lk : left knee

    rh : right hip
    lh : left hip

    '''

    rightPose=None
    match option:
        case 1:
            label, problems = tpose(landmarks)
            rightPose="TPose"
        case 2:
            label, problems = vriksh(landmarks)
            rightPose="Vrikshasana"
        case 3:
            label = veera(landmarks)
            rightPose="Veerabhadrasana"
        case 4:
            label, problems = vajrasana(landmarks)
            rightPose="Vajrasana"       
        case 5:
            label, problems = namaskara(landmarks)
            rightPose="Namaskara"
        case 6:
            label, problems = downdog(landmarks)
            rightPose="Downward-Dog"

    
    if label != 'Unknown':
        labelColor = (0, 255, 0)

    ## Print name of asana
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, labelColor, 2)
    
    flag=False
    if not problems:
        problems=["Perfect!!!"]
        flag=True
        problemColor = (255, 255, 0)

    count=0
    for i in problems:
        count+=25
    #Print problems
        cv2.putText(output_image, i, (10, 100+count), cv2.FONT_HERSHEY_PLAIN, 1, problemColor, 2)

    if display:

        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);plt.title("Output image");plt.axis('off');

    else:
        return output_image, label, problems, rightPose, flag


def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    ### Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        # Return the output image and the found landmarks.
        return output_image, landmarks