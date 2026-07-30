# Travel Trends Recommendation System

## Abstract

Social media has become a key source of travel-related information, reflecting real-time user experiences and emerging destination trends; however, extracting reliable insights from unstructured and rapidly changing content remains challenging due to informal language and data volatility.

This study presents a dynamic travel trend recommendation system that treats social media posts as a continuously evolving dataset and transforms them into actionable recommendations using a modular, offline-first architecture. An LLM-based extraction pipeline, supported by a deterministic NER fallback, converts noisy captions into structured points of interest (POIs) stored in a normalized relational database, enabling controlled and reproducible evaluation through schema-based data simulation. A composite ranking mechanism combines audience-normalized engagement metrics with recency-aware weighting to surface current and meaningful trends. User interaction is supported through a lightweight query interface operating exclusively on pre-extracted data, ensuring consistently low-latency responses.

Experimental results show that although LLM-based extraction incurs higher offline latency than classical NER methods, it provides superior semantic coverage and reliability while maintaining user-facing response times well below defined performance thresholds. The proposed system is an end-to-end engineering solution that automates discovery, preserves data freshness, and reduces manual effort in travel planning.

## Project Layout

- Core runtime: `app.py`, `models.py`, `load_data.py`, `extract_poi.py`, `run_pipeline.py`, `fuzzy_matching.py`
- Raw data: `data/`
- Experiment scripts: `experiments/`
- Evaluation datasets and outputs: `tests/evaluation_data/`
- Benchmark outputs: `results/benchmarks/`
- SQL schema/bootstrap: `db/`
- Report files: `docs/`
- Frontend template: `html5up-solid-state/`

See `README_STRUCTURE.md` for a more detailed structure note.

## Setup

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure PostgreSQL is running and configure environment variables.

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/travel_trends
QWEN_API_KEY=<your_key>
```

`QWEN_API_KEY` is optional. If omitted, the pipeline falls back to spaCy-based extraction.

## Run

1. Start the web app:

```bash
python app.py
```

2. Open in browser:

```text
http://127.0.0.1:5000/
```

3. Run ingestion + extraction pipeline:

```bash
python run_pipeline.py
```

## Experiments

- Accuracy evaluation:

```bash
python experiments/llm_accuracy_test.py
```

- Fuzzy matching test:

```bash
python experiments/test_fuzzy_matching.py
```

- Performance benchmark:

```bash
python experiments/benchmark_performance.py
```

Outputs are written to `tests/evaluation_data/` and `results/benchmarks/`.
