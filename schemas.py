from pydantic import BaseModel


class RecommendedBook(BaseModel):
    id: str
    crop_idx: int

    title: str
    author: str
    isbn13: str

    description: str
    thumbnail: str

    query_score: float

    corners: list[list[float]]


class ScanResponse(BaseModel):
    scan_id: str

    image_width: int
    image_height: int

    recommendations: list[RecommendedBook]