"""Every number the assistant uses, in one place, each with its source.

Labels used below (the same labels as the architecture plan):
  CITED     taken from a paper or official documentation (reference given)
  OPTIMUS   taken from the OptimUS code or data in this repository
  DESIGN    a design choice, with the reason given
  PILOT     a starting value only; it is fixed from the pilot run
            (20 questions + 10 script requests) before the main evaluation
  MEASURED  measured on this repository
"""

# ---------------------------------------------------------------- tokens
# MEASURED: the 24 notebooks hold 146,294 characters of markdown and code
# (outputs dropped) and 39,745 tokens with the legacy Claude tokenizer
# (tools/token_budget/measured_corpus.json on branch
# claude/project-thread-f7qjac), i.e. 3.68 characters per token.
CHARS_PER_TOKEN_LEGACY = 146294 / 39745
# CITED (approximate): Anthropic states newer models use a new tokenizer
# that produces more tokens for the same text; the token budget work used
# an uplift of 1.3. Only used for chunk sizing. Billing numbers always
# come from the API's own `usage` field (see tokenlog.py), never from this.
TOKENIZER_UPLIFT = 1.3

# ---------------------------------------------------------------- chunking
# PILOT: start at 512 tokens per piece (corpus average 384 at this target,
# token cost model, 2 Nov 2026); the pilot sweeps piece size.
CHUNK_TARGET_TOKENS = 512

# ---------------------------------------------------------------- retrieval
# CITED: Robertson & Zaragoza (2009), "The Probabilistic Relevance
# Framework: BM25 and Beyond", usual defaults k1 in [1.2, 2.0], b = 0.75.
BM25_K1 = 1.2
BM25_B = 0.75
# CITED: Cormack, Clarke & Buettcher (2009), Reciprocal Rank Fusion, k = 60.
RRF_K = 60
# PILOT: pieces sent to the model. Sweep 1, 3, 5, 8, 10, 20; best k is the
# smallest k inside the 95% Wilson interval of the best (criterion R2).
TOP_K = 8
TOP_K_SWEEP = (1, 3, 5, 8, 10, 20)

# ---------------------------------------------------------------- routing
# PILOT: cosine similarity for a "close match" (route 2) and for the
# similarity cache. Start value from GPT Semantic Cache (Regmi & Pun 2024,
# arXiv 2411.05276), where 0.8 gave 92.5-97.3% correct hits.
CLOSE_MATCH_COSINE = 0.8
# PILOT: a BM25 top score below this means "weak match" and the local
# answer falls back to Haiku. Placeholder until the pilot sets it.
WEAK_MATCH_BM25 = 5.0

# ---------------------------------------------------------------- debugging loop
# CITED: Adnan & Kuhn (2025), Debugging Decay Index, arXiv 2506.18403:
# most models lose 60-80% of debugging ability within 2-3 attempts and a
# fresh restart beats a long patch chain.
MAX_FIXES = 2
MAX_FRESH_RESTARTS = 1

# ---------------------------------------------------------------- mesh check
# CITED: "six elements per wavelength" rule, examined in Marburg (2002),
# J. Comput. Acoust. 10(1) 25-51. Success criterion S3: fail below 4,
# warn between 4 and 6.
ELEMENTS_PER_WAVELENGTH_FAIL = 4.0
ELEMENTS_PER_WAVELENGTH_WARN = 6.0
# OPTIMUS: water, row 114 of optimus/material/Material_database.xlsx.
DEFAULT_SPEED_OF_SOUND = 1482.3

# ---------------------------------------------------------------- running scripts
# OPTIMUS: the solver only runs in this image (Python 3.5 + bempp-legacy).
DOCKER_IMAGE = "optimuslib/optimus:latest"
# DESIGN: success criterion N2, under 5 minutes per notebook.
RUN_TIMEOUT_S = 300

# ---------------------------------------------------------------- models
# DESIGN: hybrid plan of 3 Nov 2026. Names are Anthropic API model ids and
# Ollama tags; change them here only.
MODEL_QUESTIONS_FALLBACK = "claude-haiku-5-5"
MODEL_SCRIPTS = "claude-sonnet-5-5"
MODEL_LOCAL = "qwen3.5:9b"
MODEL_LOCAL_EMBED = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
