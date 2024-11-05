# Knowledge-Query-System

 This Knowledge Query System is designed as a powerful multi-source retrieval
 platform, allowing users to access information from Arxiv, Stack Exchange,
 Wikipedia, and a vector store of domain-specific documents. Built with
 LangChain As the core framework, this project employs its robust Utility API
 Wrappers to seamlessly connect with external data sources, facilitating direct
 querying for academic research, technical Q&A, and general knowledge. A
 Cassandra vector store serves as a repository for focused content on AI and
 prompt engineering, enabling advanced similarity-based retrieval through
 Hugging Face embeddings. The system utilizes LangChain’s
 WebBaseLoader for efficient content extraction, paired with a dynamic routing
 model that leverages ChatGroq to determine the most relevant source for each
 user query, enhancing response accuracy.

 The user interface is developed with Streamlit, providing a responsive and
 interactive layout where users can enter queries, view results with source tags,
 and monitor query statistics. The integration of Pandas for data handling in the UI
 and tqdm for progress tracking during document processing enables smooth,
 efficient operations. Arxiv, Wikipedia, and StackAPI wrappers enable
 streamlined access to these popular platforms, and Pydantic aids in data
 validation, ensuring system stability. This combination of cutting-edge libraries
 and tools makes the Knowledge Query System an intuitive, versatile, and reliable
 solution for multi-source information retrieval.
