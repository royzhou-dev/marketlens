import google.generativeai as genai
from google.genai import types as genai_types
from google import genai as genai_new
from datetime import datetime, timedelta
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_CONTEXT_LENGTH


class GeminiClient:
    """Handles interactions with Google Gemini API"""

    SYSTEM_PROMPT = """#Context
    You are an expert stock market analyst with access to the following data for a particular stock ticker:
    - Real-time stock data (price, volume, market cap)
    - Historical price charts and trends
    - News articles with full content
    - Dividend and split history

    # Task
    Your task is answer questions related to the ticker based on the information you have been provided.

    When answering questions:
    - Be concise and data-driven
    - Cite specific numbers from the data provided
    - Format numbers with proper units (e.g., $1.5B, 10.5M shares)
    - Do not include any markdown syntax elements. There is no need to format plain text.

    Always ground your responses in the provided data. If information is not available, say so clearly."""

    def __init__(self, api_key=None, model=None):
        genai.configure(api_key=api_key or GEMINI_API_KEY)
        self.model_name = model or GEMINI_MODEL
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.SYSTEM_PROMPT
        )

    def generate_response(self, prompt, conversation_history=None):
        """
        Generate a response from Gemini

        Args:
            prompt: User prompt with context
            conversation_history: List of previous messages

        Returns:
            Generated response text
        """
        try:
            # Convert history to Gemini format and start chat
            chat = self.model.start_chat(
                history=self._convert_history(conversation_history)
            )

            # Generate response
            response = chat.send_message(prompt)
            return response.text

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I encountered an error processing your request. Please try again."

    def stream_response(self, prompt, conversation_history=None):
        """
        Generate a streaming response from Gemini

        Args:
            prompt: User prompt with context
            conversation_history: List of previous messages

        Yields:
            Response chunks as they arrive
        """
        try:
            # Convert history to Gemini format and start chat
            chat = self.model.start_chat(
                history=self._convert_history(conversation_history)
            )

            # Stream response
            response = chat.send_message(prompt, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            print(f"Error streaming response: {e}")
            yield f"I encountered an error processing your request. Please try again."

    def _convert_history(self, history):
        """
        Convert OpenAI-style history to Gemini format

        Args:
            history: List of messages with 'role' and 'content' keys

        Returns:
            List of Gemini-formatted messages
        """
        if not history:
            return []

        gemini_history = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Gemini uses 'user' and 'model' roles
            gemini_role = 'user' if role == 'user' else 'model'

            gemini_history.append({
                'role': gemini_role,
                'parts': [content]
            })

        return gemini_history


class ConversationManager:
    """Manages conversation history for chat sessions"""

    def __init__(self):
        self.conversations = {}
        self.ttl_hours = 24

    def add_message(self, conversation_id, role, content):
        """
        Add a message to conversation history

        Args:
            conversation_id: Unique conversation identifier
            role: Message role (user/assistant)
            content: Message content
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'messages': [],
                'created_at': datetime.now()
            }

        self.conversations[conversation_id]['messages'].append({
            'role': role,
            'content': content
        })

        # Clean up old conversations
        self._cleanup_old_conversations()

    def get_history(self, conversation_id, last_n=5):
        """
        Get conversation history

        Args:
            conversation_id: Unique conversation identifier
            last_n: Number of recent exchanges to return

        Returns:
            List of recent messages
        """
        if conversation_id not in self.conversations:
            return []

        messages = self.conversations[conversation_id]['messages']

        # Return last N exchanges (user + assistant pairs)
        return messages[-(last_n * 2):] if len(messages) > last_n * 2 else messages

    def clear_conversation(self, conversation_id):
        """Clear a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def _cleanup_old_conversations(self):
        """Remove conversations older than TTL"""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)

        to_delete = [
            conv_id for conv_id, data in self.conversations.items()
            if data['created_at'] < cutoff
        ]

        for conv_id in to_delete:
            del self.conversations[conv_id]


class AgentLLMClient:
    """Gemini client with function calling support using google-genai SDK."""

    SYSTEM_PROMPT = """You are an expert stock market analyst assistant for MarketLens.
You have access to tools that provide real-time stock data, financial statements,
news, sentiment analysis, and price forecasts.

When answering questions:
- Use your tools to fetch relevant data before answering. Do not guess prices or financial figures.
- You may call multiple tools if the question requires different types of data.
- Be concise and data-driven. Cite specific numbers from tool results.
- Format numbers with proper units (e.g., $1.5B, 10.5M shares).
- If a tool returns an error, acknowledge the issue and work with whatever data you have.
- Do not include markdown formatting syntax. Write in plain text.
- The user is currently viewing the stock ticker: {ticker}. Use this ticker for tool calls unless the user explicitly asks about a different stock.
- For general questions that do not require data (e.g., "what is a P/E ratio?"), respond directly without calling tools."""

    def __init__(self, api_key=None, model=None):
        self.client = genai_new.Client(api_key=api_key or GEMINI_API_KEY)
        self.model_name = model or GEMINI_MODEL

    def build_config(self, tools, ticker):
        """Build GenerateContentConfig with tools and system prompt."""
        from agent_tools import TOOL_DECLARATIONS

        function_declarations = [
            genai_types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters_json_schema=t["parameters"],
            )
            for t in tools
        ]

        return genai_types.GenerateContentConfig(
            system_instruction=self.SYSTEM_PROMPT.format(ticker=ticker),
            tools=[genai_types.Tool(function_declarations=function_declarations)],
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(self, contents, config):
        """Send contents to Gemini and return the full response."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

    @staticmethod
    def make_user_content(text):
        """Create a user Content message."""
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    @staticmethod
    def make_tool_response(name, result):
        """Create a tool response Content message."""
        return genai_types.Content(
            role="tool",
            parts=[genai_types.Part.from_function_response(name=name, response={"result": result})],
        )

    @staticmethod
    def history_to_contents(history):
        """Convert ConversationManager history to genai Contents list."""
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])])
            )
        return contents

    @staticmethod
    def extract_parts(response):
        """Extract function_call parts and text from a response."""
        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_calls = [p for p in parts if p.function_call]
        text_parts = [p.text for p in parts if p.text]

        return function_calls, text_parts, candidate.content
