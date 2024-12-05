import json
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Define data models for symptom data
class SymptomDetails:
    def __init__(self, present=False, intensity=0, description=""):
        self.present = present
        self.intensity = intensity
        self.description = description

class SymptomData:
    def __init__(self, Anxiety=None, Sleep=None, Depression=None):
        self.Anxiety = Anxiety if Anxiety is not None else SymptomDetails()
        self.Sleep = Sleep if Sleep is not None else SymptomDetails()
        self.Depression = Depression if Depression is not None else SymptomDetails()

# Sample GAD-7, PHQ-9, and ISI questions for demonstration
GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge?",
    "Not being able to stop or control worrying?",
    "Worrying too much about different things?",
    "Trouble relaxing?",
    "Being so restless that it's hard to sit still?",
    "Becoming easily annoyed or irritable?",
    "Feeling afraid, as if something awful might happen?"
]

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself – or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite – being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead, or of hurting yourself in some way?"
]

ISI_QUESTIONS = [
    "How severe have your insomnia problems been in the past 7 days?",
    "How much difficulty do you have falling asleep?",
    "How often do you wake up during the night?",
    "How much trouble do you have staying asleep?",
    "How often do you wake up feeling unrefreshed?",
    "How much do sleep problems interfere with your daily functioning?",
    "How worried or upset do you feel about your sleep?"
]


# Function to analyze symptom progress
def analyze_symptom_progress(session1, session2, selected_symptom):
    score1 = getattr(session1, selected_symptom).intensity
    score2 = getattr(session2, selected_symptom).intensity
    description1 = getattr(session1, selected_symptom).description
    description2 = getattr(session2, selected_symptom).description
    
    if score1 > score2:
        return f"Symptom '{selected_symptom}' improved from {score1} to {score2}. In session 1, {description1} while in session 2, {description2}"
    elif score1 < score2:
        return f"Symptom '{selected_symptom}' worsened from {score1} to {score2}. Description: {description1} {description2}"
    else:
        return f"Symptom '{selected_symptom}' remained the same with a score of {score1}. Description: {description1} {description2}"

# Streamlit interface
def user_interface(llm):
    st.title("Therapy Progress Tracker")

    # Upload session notes
    session_files = st.file_uploader("Upload session notes in Text or JSON format", type=["txt", "json"], accept_multiple_files=True)
    sessions = []
    for file in session_files:
        content = file.read().decode("utf-8")
        try:
            session = json.loads(content)
            sessions.append(session)
        except json.JSONDecodeError:
            st.warning(f"File {file.name} is not a valid JSON file.")

    # Symptom selection
    symptom_keywords = ["Anxiety", "Depression", "Sleep"]
    selected_symptom = st.selectbox("Select a symptom to track", symptom_keywords)

    # Analyze progress button
    if st.button("Analyze Progress"):
        if len(sessions) < 2:
            st.warning("Please upload at least two sessions for progress analysis.")
        else:
            with st.spinner("Analyzing progress..."):
                # Determine assessment type and questions
                if selected_symptom == "Anxiety":
                    assessment_type = "GAD-7"
                    questions = GAD7_QUESTIONS
                elif selected_symptom == "Depression":
                    assessment_type = "PHQ-9"
                    questions = PHQ9_QUESTIONS
                elif selected_symptom == "Sleep":
                    assessment_type = "ISI"
                    questions = ISI_QUESTIONS
                else:
                    st.warning("Invalid symptom selected.")
                    return

                # Prompt template
                prompt_template = PromptTemplate(
                    input_variables=["session", "assessment_type", "questions"],
                    template=(
                        "Given the session notes, calculate the {assessment_type} score using the questions provided.\n"
                        "Session notes: {session}\n"
                        "Questions: {questions}\n"
                        "Provide the assessment results in the following JSON format, output the JSON and nothing else:\n"
                        "{{'Anxiety': {{'present': bool, 'intensity': int, 'description': str}}, 'Sleep': {{'present': bool, 'intensity': int, 'description': str}}, 'Depression': {{'present': bool, 'intensity': int, 'description': str}}}}\n"
                        "Ensure that 'intensity' is an integer between 0 and 10 and 'description' is a string with no more than 100 words."
                        "Very important note: Output this JSON and NOTHING ELSE."
                    )
                )

                # Generate assessment for session 1
                prompt1 = prompt_template.format(
                    session=json.dumps(sessions[0]),
                    assessment_type=assessment_type,
                    questions=questions
                )
                response1 = llm.generate([prompt1])
                assessment_response_session1 = response1.generations[0][0].text.strip()

                # Generate assessment for session 2
                prompt2 = prompt_template.format(
                    session=json.dumps(sessions[1]),
                    assessment_type=assessment_type,
                    questions=questions
                )
                response2 = llm.generate([prompt2])
                assessment_response_session2 = response2.generations[0][0].text.strip()

                # Parse JSON responses
                try:
                    data_session1 = json.loads(assessment_response_session1)
                    data_session2 = json.loads(assessment_response_session2)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format in LLM response: {e}")
                    st.write(assessment_response_session1)
                    st.write('#############################')
                    st.write(assessment_response_session2)
                    # Use default values if parsing fails
                    data_session1 = {
                        "Anxiety": {"present": False, "intensity": 0, "description": ""},
                        "Sleep": {"present": False, "intensity": 0, "description": ""},
                        "Depression": {"present": False, "intensity": 0, "description": ""}
                    }
                    data_session2 = {
                        "Anxiety": {"present": False, "intensity": 0, "description": ""},
                        "Sleep": {"present": False, "intensity": 0, "description": ""},
                        "Depression": {"present": False, "intensity": 0, "description": ""}
                    }

                # Create SymptomData instances
                session1_data = SymptomData(
                    Anxiety=SymptomDetails(**data_session1.get("Anxiety", {})),
                    Sleep=SymptomDetails(**data_session1.get("Sleep", {})),
                    Depression=SymptomDetails(**data_session1.get("Depression", {}))
                )
                session2_data = SymptomData(
                    Anxiety=SymptomDetails(**data_session2.get("Anxiety", {})),
                    Sleep=SymptomDetails(**data_session2.get("Sleep", {})),
                    Depression=SymptomDetails(**data_session2.get("Depression", {}))
                )

                # Analyze progress
                progress = analyze_symptom_progress(session1_data, session2_data, selected_symptom)
                st.write(f"Progress Analysis: {progress}")

# Main function
def main():
    llm = OllamaLLM(
        model="llama3.1:latest",
        base_url="http://localhost:11434"
    )
    user_interface(llm)

if __name__ == "__main__":
    main()