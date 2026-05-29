import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PHASE_0_JSON = OUTPUT_DIR / "phase_0" / "retail_knowledge_base.json"
PHASE_1_DIR = OUTPUT_DIR / "phase_1"
QDRANT_PATH = OUTPUT_DIR / "qdrant_storage"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

QDRANT_COLLECTION = "retail_chunks"
TOP_K = 5

FILTER_DOMAIN_MAP = {
    "store operations": "Store Operations",
    "ecommerce": "eCommerce",
    "e-commerce": "eCommerce",
    "omnichannel": "Omnichannel",
    "oms": "OMS",
    "inventory": "Inventory",
    "warehouse": "Warehouse",
    "supply chain": "Supply Chain",
    "loyalty": "Loyalty",
    "promotions": "Promotions",
    "procurement": "Procurement",
    "fulfillment": "Fulfillment",
    "return": "Return",
    "pos": "POS",
    "merchandising": "Merchandising",
    "customer": "Customer",
    "finance": "Finance",
    "faq": "FAQ",
    "training": "Training",
}
