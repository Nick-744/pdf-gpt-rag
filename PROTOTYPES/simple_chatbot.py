from llama_index.llms.huggingface import HuggingFaceLLM
from context import return_context
import torch

class SimpleContextChatbot:
    def __init__(self,
                 context:    str = '',
                 model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'):
        self.llm     = None
        self.context = context

        try:
            self._initialize(model_name)
            self.is_initialized = True
        except Exception as e:
            print(f'Initialization failed: {e}')
            self.is_initialized = False

        return;

    def _initialize(self, model_name: str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Initialize LLM --- #
        self.llm = HuggingFaceLLM(
            model_name      = model_name,
            tokenizer_name  = model_name,
            device_map      = device,
            context_window  = 4096,
            max_new_tokens  = 120,
            generate_kwargs = {
                'temperature': 0.4,
                'do_sample':   True, # Without do_sample, temperature and top_p would be ignored!
                'top_p':       0.9,

                'repetition_penalty':   1.18,
                'no_repeat_ngram_size': 3
            }
        )

        return;

    def chat(self, user_input: str) -> str:
        ''' Answer by stuffing the whole CONTEXT into the prompt. '''

        if not self.is_initialized:
            return 'Sorry, the chatbot is not initialized.';
        if not user_input.strip():
            return 'Please ask a question about the context.';

        # --- Hard filter in user queries --- #
        query    = user_input.lower()
        patterns = [
            'what time', 'time is it', 'current time', 'time now',
            'date today', "today's date", 'current date', 'day today',
            'weather', 'temperature outside', 'your name', 'who are you',
            'what is your name'
        ]
        if any(p in query for p in patterns):
            return 'Not in context.';

        prompt = f'''You are a concise QA assistant.
Use ONLY the information inside CONTEXT to answer the QUESTION.

Rules:
- No preamble, no explanations, no reasoning steps, no citations, no source mentions, no document titles.
- Do not say phrases like "according to" or "the context".
- If the answer is not present in CONTEXT, output exactly: Not in context.
- Do not fabricate.
- If asked about current time, date, weather, your identity, or anything not explicitly in CONTEXT: Not in context.

CONTEXT START
{self.context}
CONTEXT END

QUESTION: {user_input}
ANSWER: '''

        text: str = ''
        try:
            response = self.llm.complete(prompt)
            text     = getattr(response, 'text', str(response))
        except Exception as e:
            print(f'Error during chat: {e}')
            text = 'I encountered an error while processing your question...'
        
        return text;

def main():
    bot = SimpleContextChatbot(context = return_context())

    print('=' * 27)
    print('Simple Chatbot from context')
    print('=' * 27)
    while True:
        user_in = input('\n- You: ')
        if user_in.lower() in ['exit', 'quit']:
            break;
        
        print('- Chat: ', end = '', flush = True)
        out = bot.chat(user_in)
        print(out)

    return;

if __name__ == '__main__':
    main()
