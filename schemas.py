from pydantic import BaseModel


class SpineResult(BaseModel):
    text: str


class ScanResponse(BaseModel):
    scan_id: str
    spines: list[SpineResult]