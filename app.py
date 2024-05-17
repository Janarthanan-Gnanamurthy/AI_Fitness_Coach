import cv2
import mediapipe as mp
import time
import math
from tkinter import *
from PIL import Image, ImageTk


class PoseDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False,
                                     model_complexity=1,
                                     smooth_landmarks=True,
                                     enable_segmentation=False,
                                     smooth_segmentation=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle


class App:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.cap = cv2.VideoCapture(0)
        self.pTime = 0
        self.canvas = Canvas(root, width=1200, height=800)
        self.canvas.pack()

        self.exercises = {
            "Dumbbells": 0,
            "Jumps": 0,
            "Jumping Jacks": 0,
            "Twisters": 0,
            "Pushups": 0,
            "Squats": 0,
        }

        self.labels = {}
        for i, (exercise, count) in enumerate(self.exercises.items()):
            label = Label(
                root, text=f"{exercise}: {count}", font=("Helvetica", 16))
            label.place(x=10, y=30 * i + 10)
            self.labels[exercise] = label

        self.photo = None
        self.squat_state = "up"
        self.update()

    def update(self):
        success, img = self.cap.read()
        if not success:
            return
        img = cv2.resize(img, (900, 600))
        img = self.detector.findPose(img)
        lmList = self.detector.getPosition(img, draw=False)

        if lmList:
            # Ensure landmarks for squats are within frame bounds
            if all(0 <= lm[1] < img.shape[1] and 0 <= lm[2] < img.shape[0] for lm in [lmList[24], lmList[26], lmList[28]]):
                angle = self.detector.findAngle(img, 24, 26, 28)
                if angle > 160 and self.squat_state == "down":
                    self.exercises["Squats"] += 1
                    self.squat_state = "up"
                    self.labels["Squats"].config(
                        text=f"Squats: {self.exercises['Squats']}")
                elif angle < 90:
                    self.squat_state = "down"

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        # Convert image to PhotoImage
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(image=img)

        # Display image on canvas
        self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
        self.root.after(10, self.update)


if __name__ == '__main__':
    root = Tk()
    root.title("AI Personal Trainer")
    detector = PoseDetector()
    app = App(root, detector)
    root.mainloop()
