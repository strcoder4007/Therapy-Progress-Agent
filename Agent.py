import re
import json
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Union

# Define Pydantic models for symptom data
class SymptomDetails(BaseModel):
    present: bool = False
    intensity: int = 0

class SymptomData(BaseModel):
    Anxiety: SymptomDetails = SymptomDetails()
    Sleep: SymptomDetails = SymptomDetails()
    Depression: SymptomDetails = SymptomDetails()

# Create Pydantic output parser
parser = PydanticOutputParser(pydantic_object=SymptomData)
format_instructions = parser.get_format_instructions()

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

# Function to extract symptoms from the session data using LLM
def extract_symptoms_with_llm(session, symptom_keywords, llm):
    symptoms_str = ", ".join(symptom_keywords)
    
    prompt = PromptTemplate(
        input_variables=["session", "symptoms"],
        template=(
            "Given the session notes: {session}, identify the presence and intensity of the following symptoms: {symptoms}. "
            "Please provide your answer in JSON format only, without any additional text or explanation. "
            "Include all specified symptoms in the JSON response, even if they are not present, with 'present': False and 'intensity': 0. "
            "Example format: {{\"Anxiety\": {{\"present\": True, \"intensity\": 7}}, \"Sleep\": {{\"present\": False, \"intensity\": 0}}, \"Depression\": {{\"present\": True, \"intensity\": 2}} }} "
            "Ensure the output is a valid JSON object."
        )
    )
    
    response = llm(prompt.format(session=session, symptoms=symptoms_str))
    
    # Parse the structured output
    try:
        parsed_response = parser.parse(response)
        parsed_response_dict = parsed_response.model_dump()
    except Exception as e:
        st.error(f"Error parsing JSON response: {e}")
        parsed_response_dict = {symptom: {'present': False, 'intensity': 0} for symptom in symptom_keywords}
    
    # Validate and clean the symptom_data
    for symptom in symptom_keywords:
        if symptom not in parsed_response_dict:
            parsed_response_dict[symptom] = {'present': False, 'intensity': 0}
        if isinstance(parsed_response_dict[symptom]['present'], str):
            parsed_response_dict[symptom]['present'] = parsed_response_dict[symptom]['present'].lower() == 'true'
        try:
            parsed_response_dict[symptom]['intensity'] = int(parsed_response_dict[symptom]['intensity'])
            parsed_response_dict[symptom]['intensity'] = max(0, min(10, parsed_response_dict[symptom]['intensity']))
        except ValueError:
            parsed_response_dict[symptom]['intensity'] = 0
    
    return parsed_response_dict

# Function to analyze symptom progress
def analyze_symptom_progress(symptoms_data):
    progress_message = ""
    for symptom, details in symptoms_data.items():
        session1 = details.get('session1', {})
        session2 = details.get('session2', {})
        if session1.get('present') and not session2.get('present'):
            progress_message += f"Symptom '{symptom}' was present in Session 1 but not in Session 2. Possible improvement.\n"
        elif not session1.get('present') and session2.get('present'):
            progress_message += f"Symptom '{symptom}' was not mentioned in Session 1 but appeared in Session 2. Possible worsening.\n"
        elif session1.get('present') and session2.get('present'):
            if session2.get('intensity', 0) < session1.get('intensity', 0):
                progress_message += f"Symptom '{symptom}' improved from {session1['intensity']} to {session2['intensity']}.\n"
            else:
                progress_message += f"Symptom '{symptom}' remained the same or worsened.\n"
    return progress_message

# LangChain agent to assess progress using the GAD-7, PHQ-9, or ISI
def create_assessment_agent(assessment_type, llm):
    if assessment_type == "GAD-7":
        questions = GAD7_QUESTIONS
    elif assessment_type == "PHQ-9":
        questions = PHQ9_QUESTIONS
    elif assessment_type == "ISI":
        questions = ISI_QUESTIONS
    else:
        raise ValueError("Unsupported assessment type.")
    
    prompt = PromptTemplate(
        input_variables=["session", "assessment_type", "questions"],
        template=(
            "Given the session notes: {session}, calculate the {assessment_type} score using the following questions: {questions}. "
            "Provide the score as a JSON object including only Anxiety, Sleep, and Depression. "
            "Ensure the output is a valid JSON object."
        )
    )
    chain = prompt | llm
    return chain

# Streamlit interface for user input
def user_interface(llm):
    st.title("Therapy Progress Tracker")

    # Upload JSON session notes
    session_files = st.file_uploader("Upload session notes in Text or JSON format", type=["txt", "json"], accept_multiple_files=True)
    sessions = []
    for file in session_files:
        content = file.read().decode("utf-8")
        try:
            session = json.loads(content)
            sessions.append(session)
        except json.JSONDecodeError:
            st.warning(f"File {file.name} is not a valid JSON file.")

    # List of possible symptoms to track
    symptom_keywords = ["Sleep", "Anxiety", "Depression"]

    # Symptom selection
    selected_symptom = st.selectbox("Select a symptom to track", symptom_keywords)

    # Button to analyze progress
    if st.button("Analyze Progress"):
        if len(sessions) < 2:
            st.warning("Please upload at least two sessions for progress analysis.")
        else:
            with st.spinner('Analyzing progress...'):
                # Determine which assessment agent to use based on selected symptom
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
                    st.error("Invalid symptom selected.")
                    return
                
                # Create the appropriate assessment agent
                assessment_agent = create_assessment_agent(assessment_type, llm)
                
                # Assess both sessions
                try:
                    assessment_response1 = assessment_agent.invoke({"session": sessions[0], "assessment_type": assessment_type, "questions": questions})
                    assessment_response2 = assessment_agent.invoke({"session": sessions[1], "assessment_type": assessment_type, "questions": questions})
                except Exception as e:
                    st.error(f"Error running assessment agent: {e}")
                    return
                
                # Extract the scores from the assessment responses
                try:
                    score_session1 = json.loads(assessment_response1)[selected_symptom]['score']
                    score_session2 = json.loads(assessment_response2)[selected_symptom]['score']
                except KeyError:
                    st.error("Error extracting scores from assessment responses.")
                    return
                
                # Analyze progress
                if score_session2 < score_session1:
                    st.write(f"Symptom '{selected_symptom}' improved from {score_session1} to {score_session2}.")
                elif score_session2 > score_session1:
                    st.write(f"Symptom '{selected_symptom}' worsened from {score_session1} to {score_session2}.")
                else:
                    st.write(f"Symptom '{selected_symptom}' remained the same.")
                
                # Optionally, display the full assessment responses
                st.write(f"Assessment for Session 1: {assessment_response1}")
                st.write(f"Assessment for Session 2: {assessment_response2}")

# Main function to initialize the app
def main():
    llm = OllamaLLM(
        model="llama3.1:latest",
        base_url="http://localhost:11434"
    )
    
    # Start the user interface in Streamlit
    user_interface(llm)

# Run the application
if __name__ == "__main__":
    main()