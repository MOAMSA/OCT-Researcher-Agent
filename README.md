# OCT Question-Answering System

The provided code implements a **question-answering system** that combines machine learning with vector search to handle queries, particularly in the **OCT field**.

## Datasets Used

The system utilizes the **MedRAG/pubmed** dataset, which is a collection of biomedical literature sourced from [PubMed](https://pubmed.ncbi.nlm.nih.gov/). This dataset offers a vast amount of scientific content, especially useful for **medical and technical applications** such as Optical Coherence Tomography (OCT). It is filtered to focus on relevant terms like **"OCT"** and similar medical imaging terms, making it valuable for domain-specific questions.

## Embedding and Search

The code uses the **sentence-transformers/all-MiniLM-L6-v2** model to convert text into **vector embeddings**. These embeddings are stored in a **Pinecone vector database**, enabling efficient semantic similarity search.

## Model for Answer Generation

The system employs **Meta-LLaMA 3**, a large language model from the LLaMA family, designed for natural language processing tasks such as text generation and answering queries. After retrieving relevant documents using Pinecone, LLaMA 3 generates context-aware answers based on the scientific literature from PubMed.


