from pydantic import BaseModel


class SpineResult(BaseModel):
    id: str
    text: str
    corners: list[list[float]]


class ScanResponse(BaseModel):
    scan_id: str
    spines: list[SpineResult]