
import cv2
import time
import numpy as np

from hand import HandDetector
from utils.templates import Gesture
from utils.utils import two_landmark_distance
from utils.utils import calculate_angle, calculate_thumb_angle, get_finger_state
from utils.utils import map_gesture, draw_bounding_box, draw_fingertips
import paho.mqtt.publish as publish
import threading


class GestureDetector:
    THUMB_THRESH = [9, 8]
    NON_THUMB_THRESH = [8.6, 7.6, 6.6, 6.1]
    BENT_RATIO_THRESH = [0.76, 0.88, 0.85, 0.65]

    
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.8, min_tracking_confidence=0.5):
        
        self.hand_detector = HandDetector(static_image_mode,
                                          max_num_hands,
                                          min_detection_confidence,
                                          min_tracking_confidence)
    
    def check_finger_states(self, hand):
        landmarks = hand['landmarks']
        label = hand['label']
        facing = hand['facing']

        self.finger_states = [None] * 5
        joint_angles = np.zeros((5,3)) # 5 fingers and 3 angles each

        # wrist to index finger mcp
        d1 = two_landmark_distance(landmarks[0], landmarks[5])
        
        for i in range(5):
            joints = [0, 4*i+1, 4*i+2, 4*i+3, 4*i+4]
            if i == 0:
                joint_angles[i] = np.array(
                    [calculate_thumb_angle(landmarks[joints[j:j+3]], label, facing) for j in range(3)]
                )
                self.finger_states[i] = get_finger_state(joint_angles[i], self.THUMB_THRESH)
            else:
                joint_angles[i] = np.array(
                    [calculate_angle(landmarks[joints[j:j+3]]) for j in range(3)]
                )
                d2 = two_landmark_distance(landmarks[joints[1]], landmarks[joints[4]])
                self.finger_states[i] = get_finger_state(joint_angles[i], self.NON_THUMB_THRESH)
                
                if self.finger_states[i] == 0 and d2/d1 < self.BENT_RATIO_THRESH[i-1]:
                    self.finger_states[i] = 1
        
        return self.finger_states
    
    def detect_gesture(self, img, mode, draw=True):
        hands = self.hand_detector.detect_hands(img)
        self.detected_gesture = None

        if hands:
            hand = hands[-1]
            self.check_finger_states(hand)
            if draw:
                self.draw_gesture_landmarks(img)
            
            ges = Gesture(hand['label'])
            self.detected_gesture = map_gesture(ges.gestures,
                                                self.finger_states,
                                                hand['landmarks'],
                                                hand['wrist_angle'],
                                                hand['direction'],
                                                hand['boundary'])
            

        return self.detected_gesture
    
    def draw_gesture_landmarks(self, img):
        hand = self.hand_detector.decoded_hands[-1]
        self.hand_detector.draw_landmarks(img)
        draw_fingertips(hand['landmarks'], self.finger_states, img)
    
    def draw_gesture_box(self, img):
        hand = self.hand_detector.decoded_hands[-1]
        draw_bounding_box(hand['landmarks'], self.detected_gesture, img)

class MQTTClient:

    current_state = None
    commands_list = {
            'On':'on',
            'Off':'off',
            'One':'red',
            'Two':'blue',
            'Three':'green',
            'Four':'yellow'
        }
    @classmethod
    def send_msg(self,gesture):
        order = gesture.split(' ')[0]
        if self.current_state != order:
            if order in self.commands_list:
                print(order)
                try:
                    publish.single("promake/led/test", self.commands_list[order],hostname="127.0.0.1", port=1883)
                    self.current_state = order
                except:
                    pass


class VisionControl:
    CAM_W = 1280
    CAM_H = 720
    TEXT_COLOR = (243,236,27)
    mode='single'
    target_gesture='all'
    max_hands = 1 
    ptime = 0
    ctime = 0
    
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.CAM_W)
        self.cap.set(4, self.CAM_H)
        self.ges_detector = GestureDetector(max_num_hands=self.max_hands)

    def run(self):
        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            self.ges_detector.detect_gesture(img, self.mode)
            if self.ges_detector.detected_gesture:
                # print(ges_detector.detected_gesture)
                if self.ges_detector.detected_gesture:
                    self.ges_detector.draw_gesture_box(img)
                    
                    t = threading.Thread(target=MQTTClient.send_msg,args=[self.ges_detector.detected_gesture,])
                    t.start()
                    t.join()
            
            self.ctime = time.time()
            fps = 1 / (self.ctime - self.ptime)
            self.ptime = self.ctime

            cv2.putText(img, f'FPS: {int(fps)}', (50,50), 0, 0.8,
                        self.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
            
            cv2.imshow('Gesture detection', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    controller = VisionControl()
    controller.run()