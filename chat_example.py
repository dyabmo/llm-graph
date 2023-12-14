# Use OpenAI chat
# create a system message and a question
# Create a chain to make them parametrizable with a question and reusable
# Add context
# Add memory(conversation buffer)

from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

chat_llm = ChatOpenAI(
    openai_api_key = OPENAI_API_KEY)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True )

prompt = PromptTemplate(template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Chat History: {chat_history}
Context: {context}      
Question :{question}
""", input_variables=["chat_history","context","question"])

chat_chain = LLMChain(llm=chat_llm, memory=memory, prompt=prompt, verbose=True)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

while True:
    question = input(">")
    response = chat_chain.run(context=current_weather,
                          question=question)
    print(response)