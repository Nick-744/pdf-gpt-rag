from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
import torch

class PDFChatbotSimple:
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
            max_new_tokens  = 1024,
            generate_kwargs = {'temperature': 0.1, 'do_sample': True}
        )

        return;

    def chat(self, user_input: str) -> str:
        ''' Answer by stuffing the whole CONTEXT into the prompt. '''

        if not self.is_initialized:
            return 'Sorry, the chatbot is not initialized.';
        if not user_input.strip():
            return 'Please ask a question about the context.';

        prompt_template = PromptTemplate(
            template = '''\
You are a helpful assistant. Use ONLY the following context to answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
''')
        
        # Format the prompt with the actual values
        prompt = prompt_template.format(
            context = self.context, question = user_input
        )

        text: str = ''
        try:
            text = self.llm.predict(prompt)
        except Exception as e:
            print(f'Error during chat: {e}')
            text = 'I encountered an error while processing your question.'
        
        return text;

def main():
    temp = '''
# Brewster's Coffee House - Context Document

## About Brewster's Coffee House

Brewster's Coffee House is a family-owned specialty coffee shop established in 2018 in downtown Portland, Oregon. Founded by Maria and James Brewster, the shop focuses on ethically sourced, single-origin coffee beans and artisanal brewing methods.

## Location & Hours

**Address:** 1247 Oak Street, Portland, OR 97205  
**Phone:** (503) 555-BREW (2739)  
**Email:** hello@brewsterscoffee.com

**Operating Hours:**
- Monday - Friday: 6:00 AM - 8:00 PM
- Saturday: 7:00 AM - 9:00 PM  
- Sunday: 7:00 AM - 7:00 PM

## Menu & Pricing

### Coffee Drinks
- Espresso: $3.25
- Americano: $4.50
- Cappuccino: $5.25
- Latte: $5.75
- Mocha: $6.25
- Cold Brew: $4.75
- Pour Over (rotating single origins): $5.50

### Food Items
- Croissants (plain, almond, chocolate): $3.75
- Breakfast sandwich: $8.50
- Avocado toast: $9.25
- Blueberry muffin: $4.25
- Bagel with cream cheese: $5.50
- Soup of the day: $7.75

### Specialty Items
- House blend coffee beans (1 lb bag): $18.95
- Single origin beans (rotating selection, 1 lb bag): $22.95
- Brewster's branded travel mug: $15.99

## Services & Features

- Free WiFi (password: BrewTime2018)
- Laptop-friendly environment with charging stations
- Weekly coffee cupping sessions (Saturdays at 10 AM)
- Private event hosting for groups up to 25 people
- Coffee subscription service available
- Accepts cash, credit cards, and mobile payments

## Policies

- Dogs are welcome on the outdoor patio only
- No outside food or beverages permitted
- Study groups welcome, but please keep noise levels respectful
- Reservation required for private events (48-hour minimum notice)
- Refunds available within 24 hours of purchase with receipt

## Staff Information

- Maria Brewster: Owner & Head Roaster
- James Brewster: Owner & Operations Manager  
- Sarah Chen: Assistant Manager & Barista Trainer
- Alex Rivera: Lead Barista (morning shift)
- Jordan Kim: Barista (evening shift)

## Loyalty Program

Brewster's Rewards Program:
- Earn 1 point per $1 spent
- 10 points = $1 reward credit
- Birthday reward: Free drink of your choice
- Double points on Wednesdays
- Members get early access to new seasonal drinks
'''
    bot = PDFChatbotSimple(context = temp)

    print('=' * 27)
    print('Simple Chatbot from context')
    print('=' * 27)
    while True:
        q = input('\n- You: ')
        if q.lower() in ['exit', 'quit']:
            break;
        print('- Bot:', bot.chat(q))

if __name__ == '__main__':
    main()
