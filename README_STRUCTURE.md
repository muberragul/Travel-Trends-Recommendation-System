# Repository Structure

## Core Application (kept at repo root)

- `app.py` - Flask API and web routes
- `models.py` - SQLAlchemy models
- `load_data.py` - data ingestion from `data/`
- `extract_poi.py` - POI extraction pipeline
- `run_pipeline.py` - end-to-end loader + extraction runner
- `fuzzy_matching.py` - matching/merge logic
- `config.py` - configuration values

## Organized Folders

- `data/` - raw synthetic travel datasets
- `experiments/` - benchmarking and evaluation scripts
- `tests/evaluation_data/` - evaluation test sets and generated evaluation outputs
- `results/benchmarks/` - benchmark JSON outputs
- `db/` - SQL schema/bootstrap files
- `docs/` - report and documentation files
- `html5up-solid-state/` - frontend template/assets used by `app.py`

## Notes About Paths

- Benchmark outputs are written to `results/benchmarks/`.
- Evaluation outputs are written to `tests/evaluation_data/`.
