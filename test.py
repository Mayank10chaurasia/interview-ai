from dotenv import load_dotenv
load_dotenv()
import os

DB_URI = os.getenv("CHECKPOINT_POSTGRES_URI")
print("CHECKPOINT DB configured:", bool(DB_URI))