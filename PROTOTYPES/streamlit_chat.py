from simple_chatbot import SimpleContextChatbot
from context import return_context
from time import perf_counter
import streamlit as st

# Run the Streamlit app with:
# streamlit run streamlit_chat.py

st.set_page_config(page_title = 'Context Chat', page_icon = '💬')

@st.cache_resource(show_spinner = False)
def load_bot():
    ctx = return_context()
    bot = SimpleContextChatbot(context = ctx)

    return bot;

bot = load_bot()

st.title('Context Chat')

if not bot.is_initialized:
    st.error('LLM failed to initialize. Check model name or dependencies.')
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = [] # List of (user, bot)

with st.form('chat_form', clear_on_submit = True):
    user_input = st.text_input('Your question', placeholder = 'Ask something from context...')
    submitted  = st.form_submit_button('Send')

if submitted and user_input.strip():
    with st.spinner('Thinking...'):
        start_time = perf_counter()

        answer     = bot.chat(user_input.strip())

        elapsed    = perf_counter() - start_time
    st.session_state.history.append((user_input.strip(), answer, elapsed))

# Display history
for i, (u, a, elapsed) in enumerate(reversed(st.session_state.history)):
    st.markdown(f'**You:** {u}')
    st.markdown(f'**Bot:** {a}\n\n_<answered in {elapsed:.2f}s>_')
    st.markdown('---')

st.sidebar.header('Chatbot Info')
st.sidebar.write(f'Initialized: {bot.is_initialized}')
st.sidebar.write(f'Model: {bot.llm.model_name}')
