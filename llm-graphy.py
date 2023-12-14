from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(
    temperature = 0,
    model="gpt-3.5-turbo-instruct",
    openai_api_key=OPENAI_API_KEY)

template = PromptTemplate(template="""
Always output JSON with key as name of the person
Only mention historical figures
who is {name} ?
""", input_variables=["name"]) 

llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    output_parser= SimpleJsonOutputParser()
)
response = llm_chain.run(name = "mohamed")

print(response)