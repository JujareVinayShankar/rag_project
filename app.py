import time
import fitz  # PyMuPDF
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

# Allow your Vercel frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), x_api_key: str = Header(...), x_session_id: str = Header(...)):
    start_time = time.time()
    try:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=x_api_key)
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        doc_texts = [page.get_text() for page in doc]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.create_documents(doc_texts)
        vector_store = FAISS.from_documents(chunks, embedding_model)
        cache_index = faiss.IndexFlatL2(1536)
        sessions[x_session_id] = {"vector_store": vector_store, "cache_index": cache_index, "cache_data": []}
        time_taken = round(time.time() - start_time, 2)
        return {"status": "success", "pages": len(doc_texts), "time_taken": time_taken}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(req: ChatRequest, x_api_key: str = Header(...), x_session_id: str = Header(...)):
    if x_session_id not in sessions:
        raise HTTPException(status_code=400, detail="No document uploaded for this session.")
    user_session = sessions[x_session_id]
    vector_store = user_session["vector_store"]
    cache_index = user_session["cache_index"]
    cache_data = user_session["cache_data"]
    query = req.query
    is_cached = False
    answer = ""
    try:
        start_time = time.time()
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=x_api_key)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=x_api_key)
        query_embedding = embedding_model.embed_query(query)
        
        if len(cache_data) > 0:
            distances, indices = cache_index.search(np.array([query_embedding]).astype('float32'), 1)
            if distances[0][0] < 0.15:
                is_cached = True
                answer = cache_data[indices[0][0]][1]
                latency_ms = round((time.time() - start_time) * 1000, 2)
                return {"answer": answer, "cached": is_cached, "latency_ms": latency_ms}

        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question based on the following context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = llm.invoke(prompt)
        answer = response.content
        cache_index.add(np.array([query_embedding]).astype('float32'))
        cache_data.append((query, answer))
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {"answer": answer, "cached": is_cached, "latency_ms": latency_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))