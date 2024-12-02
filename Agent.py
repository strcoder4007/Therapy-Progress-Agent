import json
import re
from langgraph import Agent, Router
from streamlit import *

with open("note_template_explanation.txt") as f:
    note_template = json.load(f)


class GAD7Agent(Agent):
    def __init__(self):
        self.score_map = {
            "no anxiety": 0,
            "mild anxiety": 1-5,
            "moderate anxiety": 6-10,
            "severe anxiety": 11-14
        }

    def assess(self, symptom):
        score = re.search(r"Anxiety: (\d+)", symptom["Description"])[1]
        return self.score_map.get(score)

class PHQ9Agent(Agent):
    def __init__(self):
        self.score_map = {
            "no depression": 0,
            "mild depression": 1-5,
            "moderate depression": 6-10,
            "severe depression": 11-14
        }

    def assess(self, symptom):
        score = re.search(r"Depression: (\d+)", symptom["Description"])[1]
        return self.score_map.get(score)

# Router to select appropriate agent
class TherapyProgressRouter(Router):
    def __init__(self):
        self.agents = {
            "GAD-7": GAD7Agent(),
            "PHQ-9": PHQ9Agent()
        }

    def route(self, session_transcript):
        # Extract relevant information from session transcript
        symptoms = re.findall(r"Symptom (\d+)", session_transcript)
        for symptom in symptoms:
            agent = self.agents.get(symptom["Assessment"])
            if agent:
                return agent.assess(symptom)

# Function to calculate therapy progress
def calculate_progress(sessions):
    total_score = 0
    for session in sessions:
        score = TherapyProgressRouter().route(session["Transcript"])
        if score is not None:
            total_score += score
    return total_score / len(sessions)

# Streamlit
st.title("Therapy Progress Tracker")

# Allow therapist to select two or more sessions for progress estimation
selected_sessions = st.selectbox(
    "Select sessions",
    [session["ID"] for session in get_all_session_ids()]
)

# Get selected sessions
sessions = get_selected_sessions(selected_sessions)

# Calculate therapy progress
progress = calculate_progress(sessions)
st.write("Therapy Progress:", progress)

# Allow therapist to view detailed assessment results
assessments = st.selectbox(
    "Select assessments",
    [assessment["ID"] for assessment in get_all_assessment_ids()]
)

# Get selected assessment results
results = get_selected_results(assessments)

# Display assessment results
st.write(results)