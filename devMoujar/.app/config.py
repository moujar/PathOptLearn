MODEL             = "llama3.2:1b"
MAX_SEARCH_RESULTS = 6
MAX_CHUNK_CHARS    = 1500
TOP_CHUNKS         = 8
REQUEST_TIMEOUT    = 10
YT_MAX_RESULTS     = 4

DB = dict(
    host="localhost",
    port=5432,
    dbname="deepsearch",
    user="deepsearch",
    password="deepsearch",
)

API_BASE = "http://localhost:8000"
