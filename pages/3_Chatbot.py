import streamlit as st
from openai import OpenAI
import json

# --- Page Header & Initial Setup ---
st.header("🤖 AI Financial Co-Pilot")
st.caption("Your personal assistant for market analysis and model interpretation.")

# --- Initialize OpenAI Client ---
# It's best practice to initialize this once at the top.
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"Could not connect to OpenAI. Please add your API key to `.streamlit/secrets.toml`. Error: {e}", icon="🚨")
    st.stop()

# --- Initialize Session State ---
# We initialize the full message history and a placeholder for new prompts.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are an expert financial analyst and data scientist. You specialize in explaining quantitative trading models, backtesting results, and complex financial concepts like Sharpe Ratios in a clear and concise way. When you need to show a mathematical formula, respond with a JSON object containing two keys: 'explanation' (a string explaining the formula) and 'formula' (a string with the LaTeX code)."
            " You are assisting a user who is looking at a Bitcoin trading strategy dashboard."
        },
        {
            "role": "assistant",
            "content": "Hello! I'm Co-Pilot, your AI financial assistant. How can I help you analyze today's backtest results?"
        }
    ]
if "new_prompt" not in st.session_state:
    st.session_state.new_prompt = None
if "context" not in st.session_state:
    st.session_state.context = None

# --- Layout Definition ---
chat_col, sidebar_col = st.columns([3, 1])


# --- Sidebar Logic (Handles Input from Buttons) ---
with sidebar_col:
    with st.container(border=True):
        st.subheader("Quick Questions")
        st.markdown("Click a button to ask a pre-defined question.")
        
        # When a button is clicked, we set the prompt text in session_state
        # and call st.rerun() to trigger the main chat logic.
        if st.button("Summarize Backtest Results 📈"):
            performance_results = st.session_state.get('performance_results')
            if performance_results:
                st.session_state.new_prompt = "Please summarize the latest backtest results."
                st.session_state.context = performance_results
                st.rerun()
            else:
                st.toast("⚠️ No results found. Run a backtest first.", icon="⚠️")

        if st.button("What is a Sharpe Ratio?"):
            st.session_state.new_prompt = "What is a Sharpe Ratio and why is it important?"
            st.session_state.context = None
            st.rerun()
        
        if st.button("Explain the 'Long & Short' strategy."):
            st.session_state.new_prompt = "Explain the 'Long & Short' strategy used in this dashboard."
            st.session_state.context = None
            st.rerun()


# --- Main Chat Column (Handles Display and API Calls) ---
with chat_col:
    # 1. (MODIFIED) Display chat history inside a scrollable container
    with st.container(height=600):
        for message in st.session_state.messages:
            if message["role"] != "system":
                avatar = "🧑‍💻" if message["role"] == "user" else "🧠"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

    # 2. (MODIFIED) Check for and handle new prompts from either the chat_input or sidebar buttons
    if prompt_from_input := st.chat_input("Ask a question about the results..."):
        # A prompt from the input box overrides any from the sidebar
        st.session_state.new_prompt = prompt_from_input
        st.session_state.context = None

    # This central block processes any new prompt that has been set
    if st.session_state.new_prompt:
        prompt = st.session_state.new_prompt
        context = st.session_state.context
        
        # Clear them to prevent re-running on the next interaction
        st.session_state.new_prompt = None
        st.session_state.context = None

        # Add user message to state and immediately display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Prepare for and call the API
        with st.chat_message("assistant", avatar="🧠"):
            messages_for_api = st.session_state.messages.copy()
            if context:
                context_message = {
                    "role": "system",
                    "content": f"Use the following data to answer the user's request:\n\n{context}"
                }
                messages_for_api.insert(-1, context_message)

            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": m["role"], "content": m["content"]} for m in messages_for_api],
                stream=True,
            )
            response = st.write_stream(stream)
            formatted_content = response # Default to the raw response
            try:
                # Find the start and end of the JSON object
                start_index = response.find('{')
                end_index = response.rfind('}') + 1
                
                # Check if a valid JSON block was found
                if start_index != -1 and end_index != 0 and start_index < end_index:
                    # Extract the JSON substring
                    json_string = response[start_index:end_index]
                    data = json.loads(json_string)
                    
                    # Check if it's the format we expect
                    if isinstance(data, dict) and 'explanation' in data and 'formula' in data:
                        # If yes, format it nicely for markdown with LaTeX
                        formula_latex = data['formula'].replace('\\', '\\\\') # Ensure backslashes are escaped for markdown
                        formatted_content = f"{data['explanation']}\n\n$${formula_latex}$$"

            except (json.JSONDecodeError, TypeError):
                # If parsing fails for any reason, just use the original raw response
                pass
        # Add the complete assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to clear the chat input box and move the new messages into the scrollable container
        st.rerun()