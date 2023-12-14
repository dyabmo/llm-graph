from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
url = os.getenv('URL')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')


chat_llm = ChatOpenAI(temperature=0, openai_api_key = OPENAI_API_KEY)

graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".

If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.
Do not include any text except the generated Cypher statement.

Examples:

Find movies and genres:
MATCH (m:Movie)-[:IN_GENRE]->(g)
RETURN m.title, g.name

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template = CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema","question"])

cypher_chain = GraphCypherQAChain.from_llm(
    llm=chat_llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

cypher_chain.run("What role did Tom Hanks play in Toy Story?")