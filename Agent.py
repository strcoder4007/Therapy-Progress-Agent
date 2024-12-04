import re
import json
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List

# Define Pydantic models for symptom data
class SymptomDetails(BaseModel):
    present: bool = False
    intensity: int = 0

class SymptomData(BaseModel):
    Anxiety: SymptomDetails = SymptomDetails()
    Sleep: SymptomDetails = SymptomDetails()
    Depression: SymptomDetails = SymptomDetails()

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

# Define input model for assessment tools
class AssessmentInput(BaseModel):
    session: str
    assessment_type: str
    questions: List[str]

# Function to analyze symptom progress based on assessment scores
def analyze_symptom_progress(symptom_scores_session1, symptom_scores_session2, selected_symptom):
    progress_message = ""
    symptom = selected_symptom
    score1 = symptom_scores_session1.get(symptom, 0)
    score2 = symptom_scores_session2.get(symptom, 0)
    
    if score1 > score2:
        progress_message = f"Symptom '{symptom}' improved from {score1} to {score2}."
    elif score1 < score2:
        progress_message = f"Symptom '{symptom}' worsened from {score1} to {score2}."
    else:
        progress_message = f"Symptom '{symptom}' remained the same with a score of {score1}."
    
    return progress_message

# LangChain agent to assess progress using the GAD-7, PHQ-9, or ISI
def create_assessment_agent(assessment_type: str, llm, questions: List[str]):
    prompt = PromptTemplate(
        input_variables=["session", "assessment_type", "questions"],
        template=(
            "Given the session notes: {session}, calculate the {assessment_type} score using the following questions: {questions}. "
            "Provide the score in JSON format with keys 'Anxiety', 'Sleep', and 'Depression' and their corresponding scores."
        )
    )
    chain = prompt | llm
    return chain

# LangChain router that selects the appropriate agent based on session content
def define_router(llm, assessment_type: str, questions: List[str]):
    # Create agent for the selected assessment type
    assessment_agent = create_assessment_agent(assessment_type, llm, questions)

    tools = [
        StructuredTool(
            name=f"{assessment_type} Assessment",
            func=lambda input: assessment_agent.invoke(input),
            description=f"Useful for assessing {assessment_type.lower()} levels.",
            args_schema=AssessmentInput
        )
    ]
    
    # Initialize the router agent with a compatible agent type
    router = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True, handle_parsing_errors=True)
    return router

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
    symptom_keywords = ["Anxiety", "Depression", "Sleep"]

    # Symptom selection
    selected_symptom = st.selectbox("Select a symptom to track", symptom_keywords)

    # Button to analyze progress
    if st.button("Analyze Progress"):
        if len(sessions) < 2:
            st.warning("Please upload at least two sessions for progress analysis.")
        else:
            # Show a spinner while processing the data
            with st.spinner("Analyzing progress..."):
                # Determine assessment type and questions based on selected symptom
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

                # Define router for both sessions
                router = define_router(llm, assessment_type, questions)

                # Prepare input for assessment
                input_session1 = AssessmentInput(
                    session=json.dumps(sessions[0]),
                    assessment_type=assessment_type,
                    questions=questions
                )
                input_session2 = AssessmentInput(
                    session=json.dumps(sessions[1]),
                    assessment_type=assessment_type,
                    questions=questions
                )
                
                # Run assessment for session 1
                assessment_response_session1 = router.invoke(input_session1)
                # Run assessment for session 2
                assessment_response_session2 = router.invoke(input_session2)
                
                # Parse assessment scores
                try:
                    symptom_scores_session1 = json.loads(assessment_response_session1)
                    symptom_scores_session2 = json.loads(assessment_response_session2)
                except json.JSONDecodeError:
                    st.error("Error parsing assessment scores from LLM response.")
                    symptom_scores_session1 = {symptom: 0 for symptom in symptom_keywords}
                    symptom_scores_session2 = {symptom: 0 for symptom in symptom_keywords}
                
                # Analyze the symptom progress
                progress = analyze_symptom_progress(symptom_scores_session1, symptom_scores_session2, selected_symptom)
                st.write(f"Progress Analysis: {progress}")

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