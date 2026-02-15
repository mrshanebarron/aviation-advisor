import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_data")
APP_ENV = os.getenv("APP_ENV", "production")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
