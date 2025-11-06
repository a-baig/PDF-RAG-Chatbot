# ================= ./core/config.py ================= #

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",    # model id for Flash‑Lite version :contentReference[oaicite:3]{index=3}
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)