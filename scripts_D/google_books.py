import requests
import time
from rapidfuzz import fuzz

SPINOFF_KEYWORDS = [
    "journal",
    "workbook",
    "summary",
    "companion",
    "study guide",
    "notebook",
]


def is_likely_spinoff(title):

    if not title:
        return False

    title = title.lower()

    return any(keyword in title for keyword in SPINOFF_KEYWORDS)


def google_books_search(query, api_key, max_results=5, score_cutoff=60, max_retries=3, timeout=5):

    url = "https://www.googleapis.com/books/v1/volumes"

    params = {
        "q": query,
        "maxResults": max_results,
        "key": api_key
    }

    response = None

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            break

        except requests.exceptions.Timeout:

            print(
                f"[Google Books] Timeout (attempt {attempt}/{max_retries}) "
                f"query='{query}'"
            )

        except requests.exceptions.HTTPError:

            status_code = (
                response.status_code
                if response is not None
                else None
            )

            if status_code in (500, 502, 503, 504):

                wait_time = 1.5 * attempt

                print(
                    f"[Google Books] HTTP {status_code} "
                    f"(attempt {attempt}/{max_retries}) "
                    f"query='{query}' -> retrying in {wait_time:.1f}s"
                )

                time.sleep(wait_time)
                continue

            print(
                f"[Google Books] Non-retryable HTTP {status_code} "
                f"query='{query}' -> giving up"
            )

            return []

        except requests.exceptions.RequestException as e:

            print(f"[Google Books] Request failed query='{query}': {e}")
            return []

    else:

        print(
            f"[Google Books] Max retries ({max_retries}) exceeded "
            f"query='{query}'"
        )

        return []

    data = response.json()

    if "items" not in data:
        return []

    results = []

    for item in data["items"]:

        info = item.get("volumeInfo", {})

        title = info.get("title", "")
        authors = ", ".join(info.get("authors", []))
        description = info.get("description", "")

        candidate = f"{title} {authors}"

        score = fuzz.token_set_ratio(
            query.lower(),
            candidate.lower()
        )

        spinoff = is_likely_spinoff(title)
        has_description = bool(description)
        passes_score = score >= score_cutoff

        would_be_accepted = passes_score and not spinoff

        results.append({

            "Input Query": query,
            "Book Title": title,
            "Author": authors,
            "Matching Score": round(score, 2),
            "Flagged Spinoff": spinoff,
            "Has Description": has_description,
            "Would Be Accepted": would_be_accepted,

        })

    results.sort(key=lambda x: x["Matching Score"], reverse=True)

    return results