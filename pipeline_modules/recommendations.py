"""Chroma-based book recommendations."""

import numpy as np
import pandas as pd


def get_recommendations(isbn13, collection, recommendation_top_k):
    """Retrieve nearest books using a matched book's Chroma embedding."""
    k = recommendation_top_k

    if not isbn13:
        return []

    try:
        record = collection.get(
            ids=[isbn13],
            include=["embeddings"],
        )
    except Exception:
        return []

    embeddings = record.get("embeddings")

    if embeddings is None or len(embeddings) == 0:
        return []

    result = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=k + 1,
    )

    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    recommendations = []

    for rec_id, distance, metadata in zip(
        ids, distances, metadatas
    ):
        if rec_id == isbn13:
            continue

        recommendations.append({
            "ISBN13": rec_id,
            "Title": metadata.get("title", ""),
            "Author": metadata.get("authors", ""),
            "Categories": metadata.get("categories", ""),
            "Average Rating": metadata.get("average_rating", ""),
            "Thumbnail": metadata.get("thumbnail", ""),
            "Distance": distance,
        })

        if len(recommendations) >= k:
            break

    return recommendations


def _deduplicate_shelf_books(matched_books_df):
    """
    Remove duplicate shelf-book records by ISBN13.

    If the same ISBN13 appears from Local Catalog and Google Books,
    Local Catalog is preferred.

    Records without a valid ISBN13 are retained because they cannot
    be safely identified as duplicates.
    """
    if matched_books_df is None or matched_books_df.empty:
        return matched_books_df

    df = matched_books_df.copy()

    if "ISBN13" not in df.columns:
        return df

    if "Source" in df.columns:
        df["_source_priority"] = (
            df["Source"]
            .astype(str)
            .str.strip()
            .eq("Local Catalog")
            .astype(int)
        )
        df = df.sort_values(
            "_source_priority",
            ascending=False,
            kind="stable",
        )

    isbn_text = df["ISBN13"].astype(str).str.strip()

    valid_isbn = (
        isbn_text.ne("")
        & isbn_text.str.lower().ne("nan")
        & isbn_text.str.lower().ne("none")
    )

    with_isbn = df.loc[valid_isbn].drop_duplicates(
        subset=["ISBN13"],
        keep="first",
    )

    without_isbn = df.loc[~valid_isbn]

    df = pd.concat(
        [with_isbn, without_isbn],
        axis=0,
    ).sort_index(kind="stable")

    if "_source_priority" in df.columns:
        df = df.drop(columns=["_source_priority"])

    return df.reset_index(drop=True)


def recommend_from_shelf(
    matched_books_df,
    query,
    collection,
    embedding_model,
    recommendation_query_score_cutoff=None,
    top_k=None,
):
    """
    Rank shelf books against a user query using cosine similarity.

    Duplicate books are removed by ISBN13 before final ranking.
    If the same ISBN13 exists from Local Catalog and Google Books,
    the Local Catalog record is retained.

    The recommendation cutoff is applied before Query Rank and top_k.
    Therefore duplicates and below-cutoff books cannot consume
    recommendation slots.
    """
    if matched_books_df is None or matched_books_df.empty:
        return matched_books_df

    if isinstance(query, (list, tuple)):
        query = " ".join(
            str(q) for q in query if q
        )

    query = (query or "").strip()

    if not query:
        raise ValueError(
            "recommend_from_shelf requires a non-empty query"
        )

    matched_books_df = _deduplicate_shelf_books(
        matched_books_df
    )

    if matched_books_df.empty:
        return matched_books_df

    isbn_list = (
        matched_books_df["ISBN13"]
        .astype(str)
        .tolist()
    )

    try:
        records = collection.get(
            ids=isbn_list,
            include=["embeddings"],
        )
    except Exception:
        records = {
            "ids": [],
            "embeddings": [],
        }

    embedding_lookup = {
        isbn13: embedding
        for isbn13, embedding in zip(
            records.get("ids", []),
            records.get("embeddings", []),
        )
    }

    query_embedding = np.asarray(
        embedding_model.encode([query])[0]
    )

    query_norm = np.linalg.norm(query_embedding)
    scores = []

    for isbn13, description in zip(
        isbn_list,
        matched_books_df.get(
            "Description",
            [""] * len(isbn_list),
        ),
    ):
        embedding = embedding_lookup.get(isbn13)

        if (
            not isinstance(description, str)
            or not description.strip()
        ):
            description = None

        if embedding is None and description:
            embedding = embedding_model.encode(
                [description]
            )[0]

        if embedding is None:
            scores.append(-1.0)
            continue

        embedding = np.asarray(embedding)

        denom = (
            query_norm
            * np.linalg.norm(embedding)
        )

        similarity = (
            float(
                np.dot(
                    query_embedding,
                    embedding,
                ) / denom
            )
            if denom
            else -1.0
        )

        scores.append(similarity)

    ranked_df = matched_books_df.copy()
    ranked_df["Query Score"] = scores

    ranked_df = ranked_df.sort_values(
        "Query Score",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    if recommendation_query_score_cutoff is not None:
        ranked_df = ranked_df[
            ranked_df["Query Score"]
            >= recommendation_query_score_cutoff
        ].reset_index(drop=True)

    ranked_df["Query Rank"] = (
        ranked_df.index + 1
    )

    if top_k is not None:
        ranked_df = ranked_df.head(
            top_k
        ).reset_index(drop=True)

    return ranked_df