import cv2 as cv
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv.VideoCapture(0)

# Setup pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0] 
max_vol = vol_range[1]  

# Initial volume value
vol = 0
vol_bar = 400
vol_percent = 0

while True:
    success, img = cap.read()
    img = cv.flip(img, 1) 
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Get coordinates of thumb and index finger tips
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]

            # Draw visual markers
            cv.circle(img, (x1, y1), 10, (255, 0, 0), -1)
            cv.circle(img, (x2, y2), 10, (255, 0, 0), -1)
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv.circle(img, (mid_x, mid_y), 8, (0, 0, 255), -1)

            # Calculate distance between fingers
            length = np.hypot(x2 - x1, y2 - y1)

            # Convert distance to volume
            vol = np.interp(length, [20, 150], [min_vol, max_vol])
            vol_bar = np.interp(length, [20, 150], [400, 150])
            vol_percent = np.interp(length, [20, 150], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw volume bar
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
    cv.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
    cv.putText(img, f'{int(vol_percent)} %', (40, 430), cv.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # Display window
    cv.imshow("Hand Gesture Volume Control", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
