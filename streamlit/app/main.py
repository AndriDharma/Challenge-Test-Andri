import streamlit as st
import requests
import uuid
import json
from google.cloud import secretmanager
import os
# --- Page Configuration ---
# Set the title and icon for the browser tab
st.set_page_config(page_title="AI Assistant", page_icon="🤖")

# --- App Title and Description ---
st.title("🤖 AI Assistant")
st.caption("A Streamlit chatbot front-end for your internal Q&A database.")

# --- API Configuration ---
# The target URL for the chatbot API.
# Note: We don't need the CORS proxy here because the request is made from the server-side (Streamlit), not a browser.
PROJECT_ID = os.environ.get("PROJECT_ID")
SECRET_ID_DB = os.environ.get('SECRET_ID_DB')
@st.cache_resource
def access_secret():
    # secret manager
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_DB}/versions/latest"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    db_secret = json.loads(payload)
    return db_secret

db_secret = access_secret()

# --- Session State Initialization ---
# Streamlit's session_state is used to persist variables across user interactions.

# Initialize a unique session ID for the user if it doesn't already exist.
# This is equivalent to the `sessionId = crypto.randomUUID()` in the JavaScript code.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize the chat message history if it doesn't already exist.
# We'll store a list of dictionaries, where each dictionary represents a message.
if "messages" not in st.session_state:
    # Start with a welcome message from the assistant.
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with internal data today?"}
    ]

# Initialize feedback
if "get_feedback" not in st.session_state:
    st.session_state.get_feedback = False

if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""  # Initialize feedback text

# --- Display Chat History ---
# Iterate over the messages stored in the session state and display them.
# This is equivalent to the `appendMessage` function that adds divs to the chat window.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and API Call ---
# `st.chat_input` creates a text input field at the bottom of the screen.
# When the user enters a message and hits Enter, the code inside the `if` block runs.
if prompt := st.chat_input("Type your question here..."):
    
    # 1. Append and display the user's message to the chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare for the AI's response and show a loading indicator.
    # This is equivalent to `setLoading(true)` and `showTypingIndicator(true)`.
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            try:
                # 3. Prepare the payload for the API request.
                payload = {
                    "session_id": st.session_state.session_id,
                    "user_input": prompt
                }
                
                # 4. Send the message to the API.
                response = requests.post(db_secret["API_CHAT"], json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                # 5. Get the AI's answer from the JSON response.
                data = response.json()
                ai_answer = data.get("ai_answer", "Sorry, I received an unexpected response format.")

            except requests.exceptions.RequestException as e:
                # Handle connection errors, timeouts, etc.
                ai_answer = f"Sorry, I'm having trouble connecting to the server. Please try again later. (Error: {e})"
            except Exception as e:
                # Handle other potential errors, like JSON parsing issues.
                ai_answer = f"An unexpected error occurred. (Error: {e})"

            # 6. Display the AI's response.
            st.markdown(ai_answer)

    # 7. Append the AI's response to the chat history for persistence.
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
    st.session_state.get_feedback = True

# feedback code
if st.session_state.get_feedback:
    st.feedback("thumbs", key="feedback")  # No on_change here initially
    if st.session_state.feedback == 0:
        # Text area is better for feedback, and we load/save the value
        st.session_state.feedback_text = st.text_area("Write your feedback here", value=st.session_state.feedback_text, placeholder="feedback")
    if (st.session_state.feedback == 0) or (st.session_state.feedback ==1):
        if st.button("Submit Feedback"):
            if st.session_state.feedback == 1: # 1 means thumbs up
                print("sudah baik")
            elif st.session_state.feedback == 0: # 0 means thumbs down
                print(f"terdapat feedback: {st.session_state.feedback_text}")
            payload_feedback = {
            "session_id": st.session_state.session_id,
            "feedback_good_or_not": st.session_state.feedback,
            "feedback_text": st.session_state.feedback_text
            }
            response = requests.post(db_secret["API_FEEDBACK"], json=payload_feedback)
            response.raise_for_status()
            st.session_state.get_feedback = False
            st.success("Feedback submitted!") # Display success message        
