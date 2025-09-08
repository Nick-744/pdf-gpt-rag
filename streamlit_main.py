from time import perf_counter
import streamlit as st

# --- RAG Chatbot --- #
from PDF_GPT.chatbot import PDFChatbot
from PDF_GPT.config import RAGConfig
from main import _setup_pdf_path

# -- Simple Chatbot --- #
from PROTOTYPES.simple_chatbot import SimpleContextChatbot
from PROTOTYPES.context import return_context

# Run the Streamlit app with:
# streamlit run streamlit_chat.py

st.set_page_config(page_title = 'My Chat', page_icon = '💬')

@st.cache_resource(show_spinner = False)
def load_simple_bot():
    ctx = return_context()
    bot = SimpleContextChatbot(context = ctx)

    return bot;

@st.cache_resource(show_spinner = False)
def load_rag_bot():
    pdf_path = _setup_pdf_path()
    if not pdf_path:
        st.error('No PDF found in PDF_SOURCE. Add a PDF and reload.')
        st.stop()

    cfg = RAGConfig()

    return PDFChatbot(pdf_path, cfg);

# --- Choose Chatbot --- #
bot = load_rag_bot()
# bot = load_simple_bot()

st.title('My Chat')

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
try:
    st.sidebar.write(f'Model: {bot.llm.model_name}')
except AttributeError:
    st.sidebar.write(f'Model: {bot.config.model_name}')
    st.sidebar.write(f'Embed Model: {bot.config.embed_model_name}')
