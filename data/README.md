# RAG corpus

Place your analyst-report PDFs inside **`Companies-AI-Initiatives/`** so the
RAG pipeline can index them.

Expected layout:

```
data/
└── Companies-AI-Initiatives/
    ├── MSFT.pdf
    ├── NVDA.pdf
    ├── GOOGL.pdf
    └── ...
```

Alternatively, drop a `Companies-AI-Initiatives.zip` in this folder (or in the
parent directory) — the app unpacks it automatically on first run.

The PDFs shipped with the original notebook (`Companies-AI-Initiatives.zip` and
`Healthcare-AI-Initiatives.zip`) both work out of the box.
