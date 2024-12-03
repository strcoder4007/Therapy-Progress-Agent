import json
import streamlit as st
import re
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
# from langchain.llms import OpenAI
from langchain_ollama import OllamaLLM
from langchain.tools import DuckDuckGoSearchResults
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

# Helper function to score the answers to GAD-7 or PHQ-9
def calculate_score(responses, questions):
    score = sum(responses)
    return score

# Function to extract symptoms from the session data using LLM
def extract_symptoms_with_llm(session, symptom_keywords, llm):
    prompt = PromptTemplate(
        input_variables=["session", "symptoms"],
        template="Given the session notes: {session}, identify the presence and intensity of the following symptoms: {symptoms}."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(session=session, symptoms=symptom_keywords)
    # Parse response to extract symptoms and intensity
    # Placeholder for actual parsing logic
    return response

# Function to analyze symptom progress
def analyze_symptom_progress(symptoms_data):
    progress_message = ""
    for symptom, details in symptoms_data.items():
        if details['session1']['present'] and not details['session2']['present']:
            progress_message += f"Symptom '{symptom}' was present in Session 1 but not in Session 2. Possible improvement.\n"
        elif not details['session1']['present'] and details['session2']['present']:
            progress_message += f"Symptom '{symptom}' was not mentioned in Session 1 but appeared in Session 2. Possible worsening.\n"
        elif details['session1']['present'] and details['session2']['present']:
            if details['session2']['intensity'] < details['session1']['intensity']:
                progress_message += f"Symptom '{symptom}' improved from {details['session1']['intensity']} to {details['session2']['intensity']}.\n"
            else:
                progress_message += f"Symptom '{symptom}' remained the same or worsened.\n"
    return progress_message

# Define the LangChain agent to assess progress using the GAD-7 or PHQ-9
def create_assessment_agent(assessment_type, llm):
    questions = GAD7_QUESTIONS if assessment_type == "GAD-7" else PHQ9_QUESTIONS
    prompt = PromptTemplate(
        input_variables=["questions", "session"],
        template="Given the session notes and assessment questions: {questions}, calculate the {assessment_type} score."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Define the LangChain router that selects the appropriate agent based on session content
def define_router(llm):
    tools = []
    # Agent for GAD-7
    gad7_agent = create_assessment_agent("GAD-7", llm)
    tools.append(Tool(name="GAD-7 Assessment", func=gad7_agent.run, description="Useful for assessing anxiety levels."))

    # Agent for PHQ-9
    phq9_agent = create_assessment_agent("PHQ-9", llm)
    tools.append(Tool(name="PHQ-9 Assessment", func=phq9_agent.run, description="Useful for assessing depression levels."))

    router = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return router

# Streamlit interface for user input
def user_interface(llm):
    st.title("Therapy Progress Tracker")

    # Upload JSON session notes
    session_files = st.file_uploader("Upload session notes in JSON format", type=["json"], accept_multiple_files=True)
    sessions = []
    for file in session_files:
        content = file.read()
        session = json.loads(content)
        sessions.append(session)

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
            symptom_data = {}
            for symptom in symptom_keywords:
                symptom_data[symptom] = {
                    'session1': {'present': False, 'intensity': 0},
                    'session2': {'present': False, 'intensity': 0}
                }
                # Placeholder for actual intensity extraction
                # For demonstration, assume intensity decreases over sessions
                symptom_data[symptom]['session1']['intensity'] = 5
                symptom_data[symptom]['session2']['intensity'] = 3

            progress = analyze_symptom_progress(symptom_data)
            st.write(f"Progress Analysis: {progress}")

            # Select and run assessment agent
            router = define_router(llm)
            assessment_response = router.run(f"Assess therapy progress for symptom: {selected_symptom}")
            st.write(f"Assessment Result: {assessment_response}")

# Main function to initialize the app
def main():
    # Initialize OpenAI LLM
    llm = OllamaLLM(
        model="llama3.1:8b",
        base_url="http://localhost:11434"
    )
    # llm = OpenAI(temperature=0.7)
    # Start the user interface in Streamlit
    user_interface(llm)

# Run the application
if __name__ == "__main__":
    main()