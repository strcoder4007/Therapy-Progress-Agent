# Therapy Progress Tracker

This project provides an interactive web application built with Streamlit to track the progress of various symptoms (Anxiety, Depression, Sleep) over multiple therapy sessions. It uses an AI-based language model (Ollama) to analyze and compare symptom data, and generates insights into the progress of treatment.

![Web App](/images/image.png)

## Features
- Upload session notes in either JSON or text format.
- Select symptoms (Anxiety, Depression, Sleep) to track and compare progress over different sessions.
- Automatically generate symptom assessments using the GAD-7, PHQ-9, or ISI scales based on session notes.
- View the comparison of symptom progress between two uploaded sessions.

## Research process
- Found standard tests online for specific disorders. Used ISI along with GAD-7 and PHQ-9.
- Used langchain instead langgraph because I wanted only a couple of agents.
- My free OpenAI subscription was not able to handle the data, so I used Ollama and ran Meta llama3.1:8b locally on RTX GPU.

## Future Improvements
- **Enhanced NLP Models**: Implement more advanced NLP models to better understand and assess complex session notes, potentially improving the accuracy of symptom intensity scores and descriptions.
- **Visualization of Progress**: Add visualizations such as charts or graphs to better represent symptom progression, helping users (clinicians and patients) track changes over time.

## Prerequisites

Before running the app, make sure you have the following installed:
- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [Langchain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)

Ensure that your Ollama LLM server is running locally at the specified base URL (`http://localhost:11434`).

## Installation

Follow these steps to set up and run the application:

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/therapy-progress-tracker.git
cd therapy-progress-tracker
```

### 2. Set up a virtual environment 

```
python -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows
```

### 3. Install required dependencies 

```
pip install -r requirements.txt
```

### 4. Run the application

```
streamlit run app.py

```

## Example Progress Analysis Output
```
Progress Analysis: Symptom 'Anxiety' improved from 6 to 3. 
In session 1, the patient described moderate anxiety with occasional nervousness, while in session 2, they reported feeling less anxious and more in control.
```