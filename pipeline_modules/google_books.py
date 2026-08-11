"""Google Books API lookup and insertion into the local catalog."""

import hashlib
import time
import pandas as pd
import requests
from rapidfuzz import process, fuzz


def fetch_from_google_books(
    query,
    google_books_api_key,
    google_books_match_score_cutoff,
    min_description_words=0,
    normalize_text_fn=None,
    timeout=5,
    max_retries=3,
    max_results=5,
):
    """
    Search Google Books and return the first candidate passing the
    configured fuzzy validation score.
    """
    if not google_books_api_key:
        return None

    if normalize_text_fn is None:
        raise ValueError("normalize_text_fn is required")

    url = "https://www.googleapis.com/books/v1/volumes"

    params = {
        "q": query,
        "maxResults": max_results,
        "key": google_books_api_key,
    }

    response = None

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            break

        except requests.exceptions.Timeout:
            print(f"Timeout : {query}")

        except requests.exceptions.HTTPError:
            if (
                response is not None
                and response.status_code in (429, 500, 502, 503, 504)
                and attempt < max_retries - 1
            ):
                retry_after = response.headers.get("Retry-After")

                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 1.5 * (2 ** attempt)
                else:
                    delay = 1.5 * (2 ** attempt)

                if response.status_code == 429:
                    print(
                        f"429 rate limited, "
                        f"retrying in {delay:.1f}s : {query}"
                    )

                time.sleep(delay)
                continue

            return None

        except requests.exceptions.RequestException:
            return None

    else:
        return None

    data = response.json()
    items = data.get("items", [])

    if not items:
        return None

    for item in items:
        info = item.get("volumeInfo", {})

        title = info.get("title")
        authors = info.get("authors", [])

        if not title:
            continue

        candidate = normalize_text_fn(
            title + " " + " ".join(authors)
        )

        score = fuzz.token_set_ratio(
            query,
            candidate,
        )

        if score < google_books_match_score_cutoff:
            continue

        description = info.get("description") or ""

        if (
            description
            and len(description.split()) < min_description_words
        ):
            continue

        isbn13 = None

        for identifier in info.get("industryIdentifiers", []):
            if identifier["type"] == "ISBN_13":
                isbn13 = identifier["identifier"]
                break

        return {
            "Title": title,
            "Author": ", ".join(authors),
            "ISBN13": isbn13,
            "Description": description,
            "Thumbnail": (
                info.get("imageLinks", {})
                .get("thumbnail", "")
            ),
            "Source": "Google Books",
            "Match Score": float(score),
        }

    return None


def add_to_local_catalog(
    book,
    catalog_df,
    choices,
    catalog_path,
    collection,
    embedding_model,
    normalize_text_fn,
    catalog_duplicate_score_cutoff=90,
):
    """Persist a validated Google Books record and its embedding."""
    if not book:
        return catalog_df, choices

    isbn13 = book["ISBN13"]

    if not isbn13:
        source = f"{book['Title']}_{book['Author']}"
        isbn13 = "gb_" + hashlib.md5(
            source.encode()
        ).hexdigest()[:12]
        book["ISBN13"] = isbn13

    if isbn13 in catalog_df["isbn13"].values:
        return catalog_df, choices

    normalized = normalize_text_fn(
        book["Title"] + " " + book["Author"]
    )

    duplicate = process.extractOne(
        normalized,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=catalog_duplicate_score_cutoff,
    )

    if duplicate:
        return catalog_df, choices

    text_for_embedding = (
        book["Description"]
        if book["Description"]
        else f"{book['Title']} {book['Author']}"
    )

    embedding = embedding_model.encode(
        [text_for_embedding]
    ).tolist()[0]

    collection.upsert(
        ids=[isbn13],
        embeddings=[embedding],
        documents=[book["Description"]],
        metadatas=[
            {
                "title": book["Title"],
                "authors": book["Author"],
                "thumbnail": book["Thumbnail"],
                "categories": "",
                "average_rating": 0,
            }
        ],
    )

    row = {
        column: ""
        for column in catalog_df.columns
        if column not in ("match_key", "normalized_match_key")
    }

    row.update({
        "isbn13": isbn13,
        "title": book["Title"],
        "authors": book["Author"],
        "description": book["Description"],
        "thumbnail": book["Thumbnail"],
    })

    if "title_with_subtitle" in row:
        row["title_with_subtitle"] = book["Title"]

    if "tagged_description" in row:
        row["tagged_description"] = (
            isbn13 + " " + book["Description"]
        )

    catalog_df = pd.concat(
        [catalog_df, pd.DataFrame([row])],
        ignore_index=True,
    )

    key = normalize_text_fn(
        book["Title"] + " " + book["Author"]
    )

    choices.append(key)

    columns = [
        c for c in catalog_df.columns
        if c not in ("match_key", "normalized_match_key")
    ]

    pd.DataFrame([row])[columns].to_csv(
        catalog_path,
        mode="a",
        index=False,
        header=False,
    )

    return catalog_df, choices
