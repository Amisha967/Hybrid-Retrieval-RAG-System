import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_V1_URL = f"{BACKEND_URL}/api/v1"

PAGE_TITLE = "Hybrid RAG | Dense + Sparse + Cross-Encoder"
PAGE_ICON = "⚡"
