import cv2
import mediapipe as mp
import numpy as np
from uagents import Agent, Context, Model
import requests
import json
import os

# Fetch the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    raise Exception(
        "You need to provide an API key for OpenAI to use this example")

# Configuration for making requests to OpenAI
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL_ENGINE = "gpt-3.5-turbo"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

# Initialize MediaPipe Pose and OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Fetch.ai Agent
agent = Agent()


class Request(Model):
    text: str


class Error(Model):
    text: str


class Data(Model):
    value: float
    unit: str
    timestamp: str
    confidence: float
    source: str
    notes: str


def get_completion(context: str, prompt: str, max_tokens: int = 1024):
    data = {
        "model": MODEL_ENGINE,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            OPENAI_URL, headers=HEADERS, data=json.dumps(data))
        messages = response.json()['choices']
        message = messages[0]['message']['content']
    except Exception as ex:
        return None

    print("Got response from AI model: " + message)
    return message


def get_data(ctx: Context, request: str):
    context = '''    
    You are a helpful agent who can provide answers to questions along with sources and relevant context in a machine readable format.
    
    Please follow these guidelines:
    1. Try to answer the question as accurately as possible, using only reliable sources.
    2. Rate your confidence in the accuracy of your answer from 0 to 1 based on the credibility of the data publisher and how much it might have changed since the publishing date.
    3. In the last line of your response, provide the information in the exact JSON format: {"value": value, "unit": unit, "timestamp": time, "confidence": rating, "source": ref, "notes": summary}
        - value is the numerical value of the data without any commas or units
        - unit is the measurement unit of the data if applicable, or an empty string if not applicable
        - time is the approximate timestamp when this value was published in ISO 8601 format
        - rating is your confidence rating of the data from 0 to 1
        - ref is a url where the data can be found, or a citation if no url is available
        - summary is a brief justification for the confidence rating (why you are confident or not confident in the accuracy of the value)
    '''

    response = get_completion(context, request, max_tokens=2048)

    try:
        data = json.loads(response.splitlines()[-1])
        msg = Data.parse_obj(data)
        return msg
    except Exception as ex:
        ctx.logger.exception(
            f"An error occurred retrieving data from the AI model: {ex}")
        return Error(text="Sorry, I wasn't able to answer your request this time. Feel free to try again.")


@agent.on_message(model=Request)
async def handle_request(ctx: Context, sender: str, request: Request):
    ctx.logger.info(f"Got request from {sender}: {request.text}")
    response = get_data(ctx, request.text)
    await ctx.send(sender, response)


def send_motivational_message(exercise_type, count):
    request = Request(
        text=f"Motivate the user for completing {count} {exercise_type}s!")
    agent.run_in_thread(lambda ctx: handle_request(ctx, "user_agent", request))

# Function to calculate angle between three points


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to detect posture based on hip and shoulder position


def detect_posture(hip, shoulder):
    hip_y = hip[1]
    shoulder_y = shoulder[1]

    if hip_y > shoulder_y:
        return "pushup"
    else:
        return "squat"


cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

counter_squat = 0
stage_squat = None

counter_pushup = 0
stage_pushup = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for posture detection
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # Detect posture
        posture = detect_posture(hip, shoulder)

        if posture == "squat":
            # Squat logic
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate squat angle
            squat_angle = calculate_angle(hip, knee, ankle)

            # Visualize squat angle
            cv2.putText(image, str(round(squat_angle, 2)),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Squat counter logic
            if squat_angle > 160:
                stage_squat = "up"
            if squat_angle < 90 and stage_squat == 'up':
                stage_squat = "down"
                counter_squat += 1
                send_motivational_message("squat", counter_squat)

        elif posture == "pushup":
            # Pushup logic
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate pushup angle
            pushup_angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize pushup angle
            cv2.putText(image, str(round(pushup_angle, 2)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Pushup counter logic
            if pushup_angle < 90:
                stage_pushup = "down"
            if pushup_angle > 160 and stage_pushup == 'down':
                stage_pushup = "up"
                counter_pushup += 1
                send_motivational_message("pushup", counter_pushup)

        # Display counters and stages
        cv2.putText(image, 'SQUAT REPS: ' + str(counter_squat),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'SQUAT STAGE: ' + (stage_squat if stage_squat else ""),
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'PUSHUP REPS: ' + str(counter_pushup),
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'PUSHUP STAGE: ' + (stage_pushup if stage_pushup else ""),
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    except:
        pass

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('Mediapipe Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    agent.run()
