from langchain.graphs import Neo4jGraph
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url="bolt://44.212.57.243:7687",
    username="neo4j",
    password="watches-striker-flap",
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot"    
)

retrievalQA = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever = movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True
)

r = retrievalQA("A mission to the moon goes wrong")
print(r)


"""
r = movie_plot_vector.similarity_search("A movie where aliens land and attack earth.")
print(r)

----------------

graph = Neo4jGraph(
    url="bolt://44.212.57.243:7687",
    username="neo4j",
    password="watches-striker-flap"
)

print(graph.schema)
r = graph.query("MATCH (m:Movie{title: 'Toy Story'}) RETURN m")
print(r)
"""