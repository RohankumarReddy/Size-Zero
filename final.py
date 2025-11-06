import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import csv
import os
import streamlit as st

# LLM implementation
from dotenv import load_dotenv
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.prompts import PromptTemplate

# Config part
load_dotenv()

# Use Google Gemini-2.5-flash for report generation
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ---------------------- Helpers ----------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom == 0:
        return 0.0
    cosine_angle = np.dot(ab, cb) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def get_landmark_coords(landmarks, name):
    lm = landmarks[getattr(mp_pose.PoseLandmark, name).value]
    return [lm.x, lm.y, lm.z]


class AngleBuffer:
    def __init__(self, maxlen=5):
        self.buf = deque(maxlen=maxlen)

    def add(self, v):
        self.buf.append(v)
        return np.mean(self.buf) if len(self.buf) > 0 else v


# ---------------------- Exercise definitions ----------------------
EXERCISES = {
    'squat_back': {
        'display': 'Squat (Back/Front/Goblet)',
        'joints': [('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'), ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')],
        'ideal': (80, 100),
        'rep_thresholds': (70, 160),
        'notes': 'Knee angle at bottom; keep chest upright.'
    },
    'lunge_forward': {
        'display': 'Lunge (Forward/Backward/Walking)',
        'joints': [('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'), ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')],
        'ideal': (80, 100),
        'rep_thresholds': (70, 160),
        'notes': 'Front knee should not pass toes; torso upright.'
    },
    'deadlift': {
        'display': 'Deadlift (Conventional/Romanian)',
        'joints': [('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'), ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE')],
        'ideal': (30, 60),
        'rep_thresholds': (20, 90),
        'notes': 'Hip hinge; spine neutral.'
    },
    'leg_press': {
        'display': 'Leg Press (approx)',
        'joints': [('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'), ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')],
        'ideal': (80, 110),
        'rep_thresholds': (70, 160),
        'notes': 'Approximation: using knee angles; camera-facing limitations.'
    },
    'calf_raise': {
        'display': 'Calf Raise',
        'joints': [('LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_ANKLE', 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX')],
        'ideal': (10, 40),
        'rep_thresholds': (5, 80),
        'notes': 'Tracks ankle/heel lift; may be approximate depending on visibility.'
    },
    'push_up': {
        'display': 'Push-up',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (70, 100),
        'rep_thresholds': (60, 160),
        'notes': 'Elbow angle; keep body straight.'
    },
    'bench_press': {
        'display': 'Bench Press (approx)',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (70, 100),
        'rep_thresholds': (60, 160),
        'notes': 'Approximated using elbow path; requires side camera for best accuracy.'
    },
    'overhead_press': {
        'display': 'Overhead Press',
        'joints': [('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP')],
        'ideal': (160, 200),
        'rep_thresholds': (90, 170),
        'notes': 'Shoulder alignment and full lock overhead.'
    },
    'bicep_curl': {
        'display': 'Bicep Curl',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (30, 60),
        'rep_thresholds': (30, 160),
        'notes': 'Elbow flexion; keep elbow near torso.'
    },
    'tricep_dip': {
        'display': 'Tricep Dip',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (70, 100),
        'rep_thresholds': (60, 160),
        'notes': 'Torso tilt and elbow angle.'
    },
    'row': {
        'display': 'Dumbbell/Barbell Row',
        'joints': [('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'), ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE')],
        'ideal': (30, 60),
        'rep_thresholds': (20, 90),
        'notes': 'Back neutral, hinge from hips; elbow close to body.'
    },
    'lat_pulldown': {
        'display': 'Lat Pulldown / Weighted Pull-ups',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (40, 90),
        'rep_thresholds': (30, 160),
        'notes': 'Elbow path and torso stability.'
    },
    'leg_curl': {
        'display': 'Leg Curl (Seated/Lying)',
        'joints': [('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'), ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')],
        'ideal': (30, 60),
        'rep_thresholds': (20, 160),
        'notes': 'Knee flexion control.'
    },
    'shoulder_lateral_raise': {
        'display': 'Shoulder Lateral Raise',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (70, 100),
        'rep_thresholds': (40, 160),
        'notes': 'Arm angle ~90 degrees at top; spine straight.'
    },
    'chest_fly': {
        'display': 'Chest Fly (Dumbbells/Machine)',
        'joints': [('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')],
        'ideal': (10, 40),
        'rep_thresholds': (5, 160),
        'notes': 'Soft elbow bend and shoulder stability.'
    }
}
EX_ORDER = list(EXERCISES.keys())


# ---------------------- Streamlit App ----------------------
def main():
    st.set_page_config(page_title="Size-Zero",page_icon="Home workouts.svg" , layout="wide")

    # Initialize session state variables if they don't exist
    if 'tracking' not in st.session_state:
        st.session_state.tracking = False
    if 'is_stopped' not in st.session_state:
        st.session_state.is_stopped = False
    if 'rep_count' not in st.session_state:
        st.session_state.rep_count = 0
    if 'session_log' not in st.session_state:
        st.session_state.session_log = []
    if 'selected_ex_key' not in st.session_state:
        st.session_state.selected_ex_key = EX_ORDER[0]

    # Dynamic UI logic based on 'tracking' state
    if st.session_state.tracking:
        # 2-column layout for active tracking
        st.markdown("<h1 style='text-align: center;'>Size-Zero — guiding every rep, preventing every injury.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Session in progress...</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(f"**Notes:** {EXERCISES[st.session_state.selected_ex_key]['notes']}")
            
            # Use buttons to set the state and trigger rerun
            if st.button('Stop'):
                st.session_state.tracking = False
                st.session_state.is_stopped = True
                st.rerun() 
            
            st.subheader("Session Stats")
            st_rep_counter = st.empty()
            st_feedback = st.empty()
        
        with col2:
            st_image = st.empty()

        run_tracker_stream(st_image, st_rep_counter, st_feedback, st.session_state.selected_ex_key)
        
    else:
        # Centered UI for inactive state
        st.markdown("<h1 style='text-align: center;'> Size-Zero — guiding every rep, preventing every injury.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Select an exercise and click Start to begin a session.</p>", unsafe_allow_html=True)
        
        col_empty_left, col_main, col_empty_right = st.columns([1, 2, 1])

        with col_main:
            # Dropdown at center
            exercise_options = [ex['display'] for ex in EXERCISES.values()]
            selected_ex_display = st.selectbox(
                "Choose an Exercise:",
                options=exercise_options,
                key='exercise_select'
            )
            
            # Map selected display name back to its key
            selected_ex_key = EX_ORDER[exercise_options.index(selected_ex_display)]
            st.session_state.selected_ex_key = selected_ex_key
            st.info(f"**Notes:** {EXERCISES[selected_ex_key]['notes']}")
            
            if st.button(' Start', use_container_width=True):
                st.session_state.tracking = True
                st.session_state.rep_count = 0
                st.session_state.session_log = []
                st.session_state.is_stopped = False
                st.rerun()

            # Post-session buttons, only show after a session has stopped
            if st.session_state.is_stopped and st.session_state.session_log:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Download button
                log_filepath = save_log(st.session_state.session_log)
                if log_filepath and os.path.exists(log_filepath):
                    with open(log_filepath, "r") as f:
                        st.download_button(
                            label="Download Rep wise calculations",
                            data=f,
                            file_name=os.path.basename(log_filepath),
                            mime='text/csv',
                            use_container_width=True
                        )

                # Report generation button
                if st.button("Generate Workout Report ", use_container_width=True):
                    with st.spinner("Generating report..."):
                     time.sleep(2.5)
                    with st.spinner("Finalizing report..."): 
                        report = generate_report_from_log(st.session_state.session_log)
                        st.markdown("### Your Workout Report")
                        st.info(report)
            elif st.session_state.is_stopped and not st.session_state.session_log:
                st.warning("No reps were logged. Start a new session to track your workout.")

# ---------------------- Main Tracker for Streamlit ----------------------
def run_tracker_stream(st_image, st_rep_counter, st_feedback, current_key):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st_feedback.error('Error: Could not open webcam.')
        return

    buffers = {ex: [AngleBuffer(maxlen=6) for _ in EXERCISES[ex]['joints']] for ex in EXERCISES}
    start_time = time.time()
    rep_count = 0
    stage = None
    lowest_angles = [180] * 2

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
        while st.session_state.tracking and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            feedback_list = []
            angles = []
            accuracy_scores = [None] * 2

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                cfg = EXERCISES[current_key]

                for i, joint_set in enumerate(cfg['joints']):
                    try:
                        a = get_landmark_coords(landmarks, joint_set[0])
                        b = get_landmark_coords(landmarks, joint_set[1])
                        c = get_landmark_coords(landmarks, joint_set[2])
                    except Exception:
                        angles.append(0)
                        continue

                    raw_angle = calculate_angle(a[:2], b[:2], c[:2])
                    smoothed = buffers[current_key][i].add(raw_angle)
                    angles.append(smoothed)

                    pos = (int(b[0] * w), int(b[1] * h))
                    cv2.putText(image, f"{round(smoothed, 1)}", pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if smoothed < lowest_angles[i]:
                        lowest_angles[i] = smoothed

                down_thresh, up_thresh = cfg['rep_thresholds']
                
                # State machine for rep counting
                if all(a < down_thresh for a in angles):
                    stage = 'down'
                if all(a > up_thresh for a in angles) and stage == 'down':
                    stage = 'up'
                    rep_count += 1
                    
                    min_a, max_a = cfg['ideal']
                    for i, low_ang in enumerate(lowest_angles):
                        if low_ang == 180:
                            accuracy_scores[i] = None
                        elif min_a <= low_ang <= max_a:
                            accuracy_scores[i] = 100
                        else:
                            err = min(abs(min_a - low_ang), abs(max_a - low_ang))
                            accuracy_scores[i] = max(0, 100 - (err * 1.5))

                    st.session_state.rep_count = rep_count
                    st.session_state.session_log.append({
                        'time': round(time.time() - start_time, 2),
                        'exercise': cfg['display'],
                        'rep': rep_count,
                        'lowest_left': lowest_angles[0],
                        'lowest_right': lowest_angles[1],
                        'acc_left': int(accuracy_scores[0]) if accuracy_scores[0] is not None else None,
                        'acc_right': int(accuracy_scores[1]) if accuracy_scores[1] is not None else None
                    })
                    lowest_angles = [180, 180]

                min_a, max_a = cfg['ideal']
                for i, a in enumerate(angles):
                    if a == 0:
                        feedback_list.append(f"Side {i + 1}: Not visible")
                        continue
                    if current_key in ['squat_back', 'lunge_forward', 'leg_press']:
                        if a > max_a + 50:
                            feedback_list.append(f"Side {i + 1}: Stand taller")
                        elif a > max_a:
                            feedback_list.append(f"Side {i + 1}: Slightly shallow")
                        elif a < min_a - 30:
                            feedback_list.append(f"Side {i + 1}: Too deep; check form")
                        else:
                            feedback_list.append(f"Side {i + 1}: Good")
                    elif current_key in ['deadlift', 'row']:
                        if a > max_a + 20:
                            feedback_list.append(f"Side {i + 1}: Hips too high")
                        elif a < min_a - 10:
                            feedback_list.append(f"Side {i + 1}: Hips too low or rounded back")
                        else:
                            feedback_list.append(f"Side {i + 1}: Good hip hinge")
                    elif current_key in ['push_up', 'bench_press', 'tricep_dip', 'lat_pulldown']:
                        if a < min_a:
                            feedback_list.append(f"Side {i + 1}: Go lower")
                        elif a > max_a:
                            feedback_list.append(f"Side {i + 1}: Extend more")
                        else:
                            feedback_list.append(f"Side {i + 1}: Good")
                    elif current_key in ['bicep_curl', 'shoulder_lateral_raise', 'chest_fly']:
                        if a < min_a:
                            feedback_list.append(f"Side {i + 1}: Good contraction")
                        elif a > max_a:
                            feedback_list.append(f"Side {i + 1}: Keep control")
                        else:
                            feedback_list.append(f"Side {i + 1}: Good")
                    else:
                        feedback_list.append(f"Side {i + 1}: {round(a, 1)} deg")

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2)
                )
            else:
                feedback_list = ['No pose detected. Adjust camera/lighting.']

            feedback_text = ' | '.join(feedback_list[:4]) if feedback_list else ''

            st_image.image(image, channels="BGR", use_container_width=True)
            st_rep_counter.metric(label="Reps", value=st.session_state.rep_count)
            st_feedback.info(f"{feedback_text}")
        
    cap.release()
    cv2.destroyAllWindows()


def save_log(log_data):
    if not log_data:
        st.info("No reps were logged in this session.")
        return None

    os.makedirs('logs', exist_ok=True)
    fname = f'logs/session_{int(time.time())}.csv'
    with open(fname, 'w', newline='') as csvfile:
        fieldnames = ['time', 'exercise', 'rep', 'lowest_left', 'lowest_right', 'acc_left', 'acc_right']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_data:
            writer.writerow(row)
    return fname

def generate_report_from_log(log_data):
    if not log_data:
        return "No session data to generate a report."
    
    log_str = json.dumps(log_data)
    
    prompt = PromptTemplate.from_template(
        "Generate a detailed, real-world fitness report based on the following JSON workout log.Don't suggest for patient's name and use what's avilable efficintly "
        "The report should be professional and easy to read. "
        "It must include: a summary of the workout, a breakdown of performance (total reps, average accuracy), "
        "and actionable advice based on the accuracy scores. "
        "Conclude with a motivational closing. Here is the data: {log_data}"
    )
    
    chain = prompt | llm
    
    report = chain.invoke({'log_data': log_str})
    
    return report.content


if __name__ == "__main__":
    main()