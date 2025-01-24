import streamlit as st
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader  # Install this library for PDF handling
from docx import Document
import os  # For environment variables

# Extract text from uploaded file
def extract_text_from_file(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = " ".join(page.extract_text() for page in reader.pages)
        elif file.name.endswith(".docx"):
            doc = Document(file)
            text = " ".join(paragraph.text for paragraph in doc.paragraphs)
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            text = ""
        return text
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return ""

@st.cache_resource
def initialize_chatbot():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        max_output_tokens=None,
        top_k=40,
        top_p=0.95,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        google_api_key=google_api_key,
    )
    memory = ConversationSummaryMemory(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key),
        return_messages=True,
    )
    conversation = ConversationChain(llm=llm, memory=memory)
    return conversation, memory

template = """
You are an AI interviewer designed to conduct dynamic and adaptive interviews. 
Your goal is to ask insightful questions based on the job description (JD) and the candidate's resume. 

Here are your instructions:
1. Start the interview by asking a question related to the Job Description (JD), start with basic general question and gradually increase the level of questions.
2. Based on the candidate's response, ask a follow-up question that explores their skills and relevant experience for the job profile.
3. Maintain a professional tone and ensure your questions are relevant to the role described in the JD.
4. If the candidate mentions a specific project, skill, or achievement in their response, ask for more details about that topic.
5. Avoid repeating questions. Each question should build on the previous ones to simulate a natural conversation.
6. If candidate is not able to answer two questions correctly consecutively according to your knowledge related to JD then try to ask questions realted to their past experience from resume.

Here’s the context:
- Job Description (JD): {jd_text}
- Candidate Resume: {resume_text}

Current conversation:
{chat_history}

Ask the first/next question.
"""

summary_template = """
You are an AI interviewer designed to conduct dynamic interviews. 
Summarize the interview conducted for the JD and provide insights about the candidate’s knowledge and fit for the job.
Also give a score between 1 to 5 inclusive to rate the candidate, 1 being lowest and 5 highest and give the reason why the score is justified.

Here’s the context:
- Job Description (JD): {jd_text}
- Candidate Resume: {resume_text}

Current Conversation:
{chat_history}
"""

prompt = PromptTemplate(input_variables=["jd_text", "resume_text", "chat_history"], template=template)
summary_prompt = PromptTemplate(input_variables=["jd_text", "resume_text", "chat_history"], template=summary_template)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "question_count" not in st.session_state:
    st.session_state["question_count"] = 0
if "interview_started" not in st.session_state:
    st.session_state["interview_started"] = False
if "interview_finished" not in st.session_state:
    st.session_state["interview_finished"] = False

with st.sidebar.header("Provide Details"):
    jd_text = st.sidebar.text_area("Job Description (JD)", placeholder="Paste the job description here")
    uploaded_resume = st.sidebar.file_uploader(
    "Upload Your Resume (PDF, Word, or Text File)", 
    type=["pdf", "docx", "txt"]
    )
    max_questions = st.sidebar.slider("Number of Questions", min_value=1, max_value=10, value=4)
    if jd_text and uploaded_resume:
        def start_interview():
            st.session_state["interview_started"] = True
        st.sidebar.button("Start Interview", on_click=start_interview)

st.title("AI Interviewer Bot")
st.caption("This bot will ask questions based on the Job Description (JD) and your Resume.")

if uploaded_resume:
    resume_text = extract_text_from_file(uploaded_resume)
else:
    resume_text = ""


if "messages" in st.session_state:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
if st.session_state["interview_started"]:
    conversation, memory = initialize_chatbot()

    if st.session_state["interview_finished"]:
        response = conversation.predict(
                        input=summary_prompt.format(
                            jd_text=jd_text,
                            resume_text=resume_text,
                            chat_history=memory.chat_memory.messages
                        )
                    )
        with st.expander("Summary"):
            st.write(response)

    if st.session_state["question_count"] == 0:
        response = conversation.predict(
                    input=prompt.format(
                        jd_text=jd_text,
                        resume_text=resume_text,
                        chat_history=memory.chat_memory.messages
                    )
                )
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        st.session_state["question_count"] += 1
    if st.session_state["question_count"]<=max_questions:
        if user_input := st.chat_input():
            print(st.session_state["question_count"])
            st.session_state.messages.append({"role": "user", "content": user_input})
            memory.chat_memory.add_user_message(user_input)
            st.chat_message("user").write(user_input)
            if st.session_state["question_count"]<max_questions:
                response = conversation.predict(
                        input=prompt.format(
                            jd_text=jd_text,
                            resume_text=resume_text,
                            chat_history=memory.chat_memory.messages
                        )
                    )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            else:
                msg = "Oh nice, That was it from my side for this interview. Type \"exit\" to get the interview summary."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                st.session_state["interview_finished"] = True
                st.session_state["question_count"] += 1
            st.session_state["question_count"] += 1