"""
Detect hands on streams.

Usage:
    $ python3 hand.py --max_hands 2
"""

import argparse
import cv2
import mediapipe as mp
import time
import numpy as np

from utils.utils import check_hand_direction, find_boundary_lm
from utils.utils import calculate_angle, display_hand_info




# A hand detector based on mediapipe, it can detect hands and return several features of hands:
#   'label'         - handedness of hands, 'left', 'right'
#   'landmarks'     - the coordinates of 21 hand joints
#   'wrist_angle'   - angle of <index finger mcp, wrist, pinky mcp>
#   'direction'     - the direction that a hand is pointing, 'up', 'down', 'left', 'right'
#   'facing'        - the facing of hands, 'front', 'back' ('front' means the palm is facing the camera)
#   'boundary'      - the boundary joints from 'up', 'down', 'left', 'right'

class HandDetector:
    LM_COLOR = (102, 255, 255)
    LINE_COLOR = (51, 51, 51)

    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.8, min_tracking_confidence=0.7):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def detect_hands(self, img):
        self.decoded_hands = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            h, w, _ = img.shape
            num_hands = len(self.results.multi_hand_landmarks)
            self.decoded_hands = [None] * num_hands

            for i in range(num_hands):
                self.decoded_hands[i] = dict()
                lm_list = list()
                handedness = self.results.multi_handedness[i]
                hand_landmarks = self.results.multi_hand_landmarks[i]
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = int((lm.z - wrist_z) * w)
                    lm_list.append([cx, cy, cz])

                label = handedness.classification[0].label.lower()
                lm_array = np.array(lm_list)
                direction, facing = check_hand_direction(lm_array, label)
                boundary = find_boundary_lm(lm_array)
                wrist_angle_joints = lm_array[[5, 0, 17]]
                wrist_angle = calculate_angle(wrist_angle_joints)

                self.decoded_hands[i]['label'] = label
                self.decoded_hands[i]['landmarks'] = lm_array
                self.decoded_hands[i]['wrist_angle'] = wrist_angle
                self.decoded_hands[i]['direction'] = direction
                self.decoded_hands[i]['facing'] = facing
                self.decoded_hands[i]['boundary'] = boundary

        return self.decoded_hands

    def draw_landmarks(self, img):
        w = img.shape[1]
        t = int(w / 500)
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(
                                                   color=self.LM_COLOR, thickness=3*t, circle_radius=t),
                                               self.mp_drawing.DrawingSpec(color=self.LINE_COLOR, thickness=t, circle_radius=t))
                
                
