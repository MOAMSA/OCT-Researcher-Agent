<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Question-Answering System README</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1 {
            color: #333;
        }
        h2 {
            color: #555;
        }
        p {
            margin: 10px 0;
        }
        .highlight {
            background-color: #f0f8ff;
            padding: 5px;
            border-left: 3px solid #00f;
        }
    </style>
</head>
<body>
    <h1>Question-Answering System Overview</h1>
    
    <p>The provided code implements a <strong>question-answering system</strong> that integrates machine learning with vector search to handle queries, particularly in the <strong>medical field</strong>.</p>
    
    <h2>Datasets Used</h2>
    <p>The system leverages the <strong>MedRAG/pubmed</strong> dataset, which is a collection of biomedical literature sourced from <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">PubMed</a>. This dataset provides a vast array of scientific content, especially useful for <strong>medical and technical applications</strong> such as Optical Coherence Tomography (OCT).</p>
    <p>The dataset is filtered to focus on relevant terms like <strong>"OCT"</strong> and similar medical imaging terms, making it particularly valuable for addressing domain-specific questions.</p>
    
    <h2>Embedding and Search</h2>
    <p>The code uses the <strong>sentence-transformers/all-MiniLM-L6-v2</strong> model to convert text into <span class="highlight">vector embeddings</span>. These embeddings are then stored in a <strong>Pinecone vector database</strong> for efficient semantic similarity search.</p>
    
    <h2>Model for Answer Generation</h2>
    <p>The system employs <strong>Meta-LLaMA 3</strong>, a large language model from the LLaMA family, for generating responses. LLaMA 3 is specifically designed for natural language processing tasks such as text generation and query answering. After retrieving the most relevant documents using Pinecone, LLaMA 3 processes this information to generate context-aware answers based on the scientific literature from PubMed.</p>
    
    <p>Overall, the system combines document retrieval with advanced <strong>natural language processing</strong> to deliver precise, contextually relevant responses to complex queries.</p>
</body>
</html>
