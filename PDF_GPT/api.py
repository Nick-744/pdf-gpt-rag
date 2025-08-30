'''
API wrapper module for PDF RAG Chatbot System
-------------------------------------------------
Provides a simple programmatic API for integration and testing.
'''

from typing import List, Dict, Any, Optional, Union

from config import RAGConfig
from chatbot import PDFChatbot

# ---------------------------
# Programmatic API
# ---------------------------
class ChatbotAPI:
    '''A small API wrapper suitable for integration/tests.'''

    def __init__(self, pdf_path: Union[str, List[str]], config: Optional[RAGConfig] = None):
        '''
        Initialize the API with PDF path(s) and optional configuration.
        
        Args:
            pdf_path: Single PDF path string or list of PDF paths
            config: Optional RAGConfig instance, uses defaults if None
        '''
        paths        = [pdf_path] if isinstance(pdf_path, str) else list(pdf_path)
        self.config  = config or RAGConfig()
        self.chatbot = PDFChatbot(paths, self.config)

        return;

    def ask(self, question: str) -> Dict[str, Any]:
        '''
        Ask a question and return the response in a structured format.
        
        Args:
            question: The question to ask about the PDF content
            
        Returns:
            Dict with keys: success (bool), message (str), response (str)
        '''
        if not self.chatbot.is_initialized:
            return {'success': False, 'message': 'Chatbot not initialized', 'response': ''};
        try:
            resp = self.chatbot.chat(question)
            return {'success': True, 'message': 'OK', 'response': resp};
        except Exception as e:
            return {'success': False, 'message': f'Error: {e}', 'response': ''};

    def reset(self) -> Dict[str, Any]:
        '''
        Reset the conversation history.
        
        Returns:
            Dict with keys: success (bool), message (str)
        '''
        self.chatbot.reset_conversation()
        
        return {'success': True, 'message': 'Conversation reset'};

    def get_info(self) -> Dict[str, Any]:
        '''
        Get information about the loaded documents and configuration.
        
        Returns:
            Dict containing document and configuration information
        '''
        return self.chatbot.get_document_info();
