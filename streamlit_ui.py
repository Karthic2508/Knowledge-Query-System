import streamlit as st
import cassio
import pandas as pd
from stackapi import StackAPI
import arxiv
import wikipedia
from datetime import datetime
from pydantic import BaseModel
from langchain_core.pydantic_v1 import Field
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm.autonotebook import tqdm, trange
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Set page config
st.set_page_config(
    page_title="Knowledge Query System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS to improve the UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
    }
    .stTextInput>div>div>input {
        height: 3em;
    }
    .result-card {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .source-tag {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-bottom: 10px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

## connection of the ASTRA DB
ASTRA_DB_APPLICATION_TOKEN="AstraCS:lMxratEJzMFZURObbEicuxkU:fdfa5af352b4573179e6d204487c5a04306d847d14ab9378f10c31bbea672b6e"
ASTRA_DB_ID="5a4f4066-9926-42a7-8861-e1286eda3486"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

# Define URLs of documents to index and store
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from URLs and prepare for splitting
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into chunks for easier processing
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Initialize embedding model
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()

# Setup for LLM routing
class RouteQuery(BaseModel):
    datasource: str = Field(..., description="Route user questions to appropriate source")

groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)


def route_question(question: str) -> str:
    """
    Route question to appropriate search function based on the routing model.
    """
    # Define `route_prompt` as a ChatPromptTemplate with clearer guidelines
    route_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=f"""You are an expert at routing the user question - '{question}' to the appropriate datasource.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics.
    Use stack_exchange for technical programming questions or when seeking solutions to specific technical problems.
    Use arxiv for questions about research papers, academic publications, or scientific literature.
    Use wiki-search for general knowledge questions."""
            ),
        HumanMessage(content="{question}")
    ])
    print("---ROUTE QUESTION---")

    # Generate the formatted prompt
    prompt_input = route_prompt.format_prompt(question=question)

    # Invoke routing LLM with formatted prompt
    routing = structured_llm_router.invoke(prompt_input)

    # Route based on the result
    if routing.datasource == "wiki_search":
        print("---ROUTE QUESTION TO WIKI SEARCH---")
        return "wiki_search"
    elif routing.datasource == "vectorstore":
        print("---ROUTE QUESTION TO VECTORSTORE---")
        return "vectorstore"
    elif routing.datasource == "stack_exchange":
        print("---ROUTE QUESTION TO STACK EXCHANGE---")
        return "stack_exchange"
    elif routing.datasource == "arxiv":
        print("---ROUTE QUESTION TO ARXIV---")
        return "arxiv"


def format_arxiv_results(results):
    st.markdown("""
        <div class='result-card' style='background-color: #f0f7ff;'>
            <div class='source-tag' style='background-color: #0366d6; color: white;'>ArXiv</div>
    """, unsafe_allow_html=True)

    for result in results:
        st.markdown(f"**Title:** {result.title}")
        st.markdown(f"**Authors:** {', '.join(author.name for author in result.authors)}")
        st.markdown(f"**Published:** {result.published.strftime('%Y-%m-%d')}")
        st.markdown(f"**Abstract:** {result.summary[:500]}...")
        st.markdown(f"[PDF Link]({result.pdf_url})")
        st.markdown("---")


def format_stack_exchange_results(results):
    st.markdown("""
        <div class='result-card' style='background-color: #fff8f0;'>
            <div class='source-tag' style='background-color: #f48024; color: white;'>Stack Exchange</div>
    """, unsafe_allow_html=True)

    for result in results:
        st.markdown(f"**Question:** {result['title']}")
        st.markdown(f"**Score:** {result['score']}")
        st.markdown(f"[View on Stack Overflow]({result['link']})")
        st.markdown("---")


def format_vector_results(results):
    st.markdown("""<div class='result-card' style='background-color: #e8f5e9;'>""", unsafe_allow_html=True)
    st.markdown(f"<div class='source-tag' style='background-color: #388e3c; color: white;'>Vector Store</div>", unsafe_allow_html=True)

    for document in results['documents']:
        content = document.page_content if isinstance(document, Document) else document
        st.markdown(f"**Content:** {content}")
        if isinstance(document, Document) and 'url' in document.metadata:
            st.markdown(f"[Source Link]({document.metadata['url']})")
        st.markdown("---")

def format_wiki_results(results):
    st.markdown("""
        <div class='result-card' style='background-color: #f5f0ff;'>
            <div class='source-tag' style='background-color: #7157d9; color: white;'>Wikipedia</div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Summary:** {results}")
    st.markdown("---")


def retrieve(state):
    """
    Retrieve documents and eliminate duplicates.
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieve documents based on the query
    documents = retriever.invoke(question)

    # Remove duplicates based on page_content
    unique_docs = {}
    for doc in documents:
        content = doc.page_content if isinstance(doc, Document) else doc
        unique_docs[content] = doc

    # Convert unique docs back to a list of Documents
    documents = list(unique_docs.values())

    # If no documents were retrieved, add a fallback message
    if not documents:
        documents = [Document(page_content="No relevant documents found in vector store.")]

    # Return the state with unique documents
    return {"documents": documents, "question": question}


def process_query(query: str):
    """
    Process the query and display results based on routing.
    """
    with st.spinner('Processing your query...'):
        source = route_question(query)

        try:
            if source == "arxiv":
                source = "arxiv"
                search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
                results = list(search.results())
                format_arxiv_results(results)

            elif source == "stack_exchange":
                source = "stack_exchange"
                SITE = StackAPI('stackoverflow')
                results = SITE.fetch('search/advanced', q=query, sort='votes', accepted=True)
                format_stack_exchange_results(results['items'][:3])

            elif source == "vectorstore":
                source = "vectorstore"
                state = {"question": query}
                results = retrieve(state)
                format_vector_results(results)

            else:
                source = "wiki_search"
                results = wikipedia.summary(query, sentences=3)
                format_wiki_results(results)

            # Append to history
            st.session_state.history.append({'query': query, 'source': source, 'timestamp': pd.Timestamp.now()})

        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")


def main():
    # Header
    st.title("🔍 Knowledge Query System")
    st.markdown("""
        Ask questions about:
        - Research papers and academic publications
        - Programming and technical issues
        - AI agents and prompt engineering
        - General knowledge
    """)

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        query = st.text_input("Enter your query:", key="query_input")

        # Submit button
        if st.button("Search", key="search_button"):
            if query:
                process_query(query)
            else:
                st.warning("Please enter a query.")

    with col2:
        # Query history
        st.subheader("Recent Queries")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(
                history_df[['timestamp', 'query', 'source']],
                hide_index=True,
                height=300
            )
        else:
            st.info("No queries yet.")

    # Metrics
    if st.session_state.history:
        st.subheader("Query Statistics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            total_queries = len(st.session_state.history)
            st.metric("Total Queries", total_queries)

        with metrics_col2:
            source_counts = pd.Series([x['source'] for x in st.session_state.history]).value_counts()
            most_common_source = source_counts.index[0]
            st.metric("Most Used Source", most_common_source)

        with metrics_col3:
            unique_queries = len(set([x['query'] for x in st.session_state.history]))
            st.metric("Unique Queries", unique_queries)


if __name__ == "__main__":
    main()