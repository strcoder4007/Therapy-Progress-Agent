import json
import streamlit as st
import re
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import OutputParserException
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

# Sample GAD-7 and PHQ-9 questions for demonstration
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

# Function to extract symptoms from the session data using LLM
def extract_symptoms_with_llm(session, symptom_keywords, llm):
    symptoms_str = ", ".join(symptom_keywords)
    
    prompt = PromptTemplate(
        input_variables=["session", "symptoms"],
        template=(
            "Given the session notes: {session}, identify the presence and intensity of the following symptoms: {symptoms}. "
            "Please provide your answer in JSON format only, without any additional text or explanation. "
            "Include all specified symptoms in the JSON response, even if they are not present, with 'present': False and 'intensity': 0. "
            "Example format: '{{'Anxiety': {{'present': True, 'intensity': 7}}, 'Sleep': {{'present': False, 'intensity': 0}}, 'Depression': {{'present': True, 'intensity': 2}}}}'\n"
            "{format_instructions}"
        ),
        partial_variables={"format_instructions": format_instructions}
    )
    
    chain = prompt | llm
    response = chain.invoke({"session": session, "symptoms": symptoms_str})
    
    # Parse the structured output
    try:
        parsed_response = parser.parse(response)
        parsed_response_dict = parsed_response.model_dump()
    except OutputParserException as e:
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

# LangChain agent to assess progress using the GAD-7 or PHQ-9
def create_assessment_agent(assessment_type, llm):
    questions = GAD7_QUESTIONS if assessment_type == "GAD-7" else PHQ9_QUESTIONS
    prompt = PromptTemplate(
        input_variables=["session", "assessment_type", "questions"],
        template="Given the session notes: {session}, calculate the {assessment_type} score using the following questions: {questions}. Provide the score as a JSON object including only Anxiety, Sleep, and Depression.",
        partial_variables={"format_instructions": format_instructions}
    )
    chain = prompt | llm
    return chain

# LangChain router that selects the appropriate agent based on session content
def define_router(llm):
    tools = []
    # Agent for GAD-7
    gad7_agent = create_assessment_agent("GAD-7", llm)
    tools.append(Tool(
        name="GAD-7 Assessment",
        func=lambda session: gad7_agent.invoke({"session": session, "assessment_type": "GAD-7", "questions": GAD7_QUESTIONS}),
        description="Useful for assessing anxiety levels."
    ))

    # Agent for PHQ-9
    phq9_agent = create_assessment_agent("PHQ-9", llm)
    tools.append(Tool(
        name="PHQ-9 Assessment",
        func=lambda session: phq9_agent.invoke({"session": session, "assessment_type": "PHQ-9", "questions": PHQ9_QUESTIONS}),
        description="Useful for assessing depression levels."
    ))

    router = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return router

# Streamlit interface for user input
def user_interface(llm):
    st.title("Therapy Progress Tracker")

    # Upload JSON session notes
    session_files = st.file_uploader("Upload session notes in JSON format", type=["json"], accept_multiple_files=True)
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
            # Extract and analyze symptoms using LLM
            symptom_data_session1 = extract_symptoms_with_llm(sessions[0], symptom_keywords, llm)
            symptom_data_session2 = extract_symptoms_with_llm(sessions[1], symptom_keywords, llm)
            
            symptoms_data = {
                selected_symptom: {
                    'session1': symptom_data_session1.get(selected_symptom, {'present': False, 'intensity': 0}),
                    'session2': symptom_data_session2.get(selected_symptom, {'present': False, 'intensity': 0})
                }
            }
            st.write(f"SYMPTOM DATA: {symptoms_data}")
            progress = analyze_symptom_progress(symptoms_data)
            st.write(f"Progress Analysis: {progress}")

            # Select and run assessment agent
            router = define_router(llm)
            assessment_response = router.invoke(f"Assess therapy progress for symptom: {selected_symptom}")
            st.write(f"Assessment Result: {assessment_response}")

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