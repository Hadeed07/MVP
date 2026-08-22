from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_connection


router = APIRouter(
    prefix="/recommendation-query",
    tags=["recommendation"],
)

class RecommendationQuery(BaseModel):
    query: str

@router.get("", response_model=RecommendationQuery)
def get_recommendation_query():
    connection = get_connection()

    row = connection.execute(
        """
        SELECT query
        FROM recommendation_preferences
        WHERE id = 1
        """
    ).fetchone()

    connection.close()

    if row is None:
        raise HTTPException(status_code=404, detail="No recommendation query has been saved.",)

    return RecommendationQuery(query=row["query"])


@router.put("", response_model=RecommendationQuery)
def save_recommendation_query(payload: RecommendationQuery):
    query = payload.query.strip()

    print("Received query:", query)

    if not query:
        raise HTTPException(status_code=400, detail="Recommendation query cannot be empty.",)

    updated_at = datetime.now(timezone.utc).isoformat()
    connection = get_connection()
    
    connection.execute(
        """
        INSERT INTO recommendation_preferences (id, query, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(id)
        DO UPDATE SET
            query = excluded.query,
            updated_at = excluded.updated_at
        """,
        (query, updated_at),
    )

    connection.commit()
    connection.close()

    return RecommendationQuery(query=query)