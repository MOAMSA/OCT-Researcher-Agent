import langchain
from torch import cuda, bfloat16
import transformers
import pandas as pd
import time
from pinecone import ServerlessSpec
import os
from pinecone import Pinecone
from datasets import load_dataset
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"

device = 'cpu'
print(device)
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)



docs = [
    "this is one document",
    "and another document"
]

embeddings = embed_model.embed_documents(docs)

print(f"We have {len(embeddings)} doc embeddings, each with "f"a dimensionality of {len(embeddings[0])}.")


# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY') or '688360f7-77dc-4e95-8a9f-6edf73bfeacc'

# configure client
pc = Pinecone(api_key=api_key)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'llama-2-rag'

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
# view index stats
index.describe_index_stats()

print("Loading dataset")
data =  load_dataset("MedRAG/pubmed")
print("Loading dataset done")

"""data = data['train'].to_pandas()

batch_size = 32

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['id']}-{x['PMID']}" for i, x in batch.iterrows()]
    texts = [x['contents'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['contents'],
         'PMID': x['PMID'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))"""
    
# Convert dataset to pandas DataFrame
data = data['train'].to_pandas()

# Define keywords to filter
keywords = ["Optical Coherence Tomography", "OCT", "SD-OCT", "SS-OCT"]

# Filter the DataFrame to include only rows with the specified keywords
filtered_data = data[data['contents'].str.contains('|'.join(keywords), case=False, na=False)]

# Limit the filtered data to a random sample of 2000 rows
filtered_data = filtered_data.sample(n=2000, random_state=42)

# Show a random sample of the filtered dataset (optional, e.g., showing 5 random rows)
print(filtered_data.sample(5))

# Batch size for processing
batch_size = 32

# Process the filtered data in batches
for i in range(0, len(filtered_data), batch_size):
    i_end = min(len(filtered_data), i + batch_size)
    batch = filtered_data.iloc[i:i_end]
    
    # Generate unique IDs
    ids = [f"{x['id']}-{x['PMID']}" for i, x in batch.iterrows()]
    
    # Extract text contents
    texts = [x['contents'] for i, x in batch.iterrows()]
    
    # Embed the texts using the embedding model
    embeds = embed_model.embed_documents(texts)
    
    # Prepare metadata
    metadata = [
        {'text': x['contents'],
         'PMID': x['PMID'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    
    # Upsert the vectors and metadata to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))



index.describe_index_stats()



model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

device = 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    #quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")





tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)




generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.001,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


llm = HuggingFacePipeline(pipeline=generate_text)

text_field = 'text'  # field in metadata that contains text content

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

query = 'denoising methods in OCT imaging?'

vectorstore.similarity_search(
    query,  # the search query
    k=3  # returns top 3 most relevant chunks of text
)



rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)


print('llm:' + llm('denoising methods in OCT imaging?'))

print(rag_pipeline('denoising methods in OCT imaging?'))






