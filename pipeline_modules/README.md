# Modular Book Spine Pipeline

This package separates the working pipeline by responsibility while keeping
`SpinePipeline` as the controller.

## Modules

- `pipeline.py` — dependency wiring and orchestration
- `config.py` — named thresholds and tunable parameters
- `detection.py` — YOLO OBB detection, annotation and perspective crops
- `preprocessing.py` — CLAHE + mild sharpening + resize/padding
- `ocr.py` — RapidOCR
- `matching.py` — local RapidFuzz catalog matching
- `google_books.py` — Google Books fallback and catalog insertion
- `recommendations.py` — Chroma book-to-book and query-to-shelf recommendations
- `tracking.py` — per-crop progress and timing DataFrames
- `visualization.py` — notebook output
- `utils.py` — both text-normalization functions and point ordering

## Explicit dependencies

Component functions do not access `SpinePipeline.self`. Models, DataFrames,
Chroma collections, thresholds and helper functions are passed as arguments.

## Recommendation cutoff

`RECOMMENDATION_QUERY_SCORE_CUTOFF` is applied after calculating cosine
similarity. Therefore the pipeline may return fewer than three query
recommendations when fewer than three shelf books meet the cutoff.

## Important

The module contents are based on the uploaded working pipeline's functions
and behavior. The main controller preserves the working flow while making
dependencies explicit.
