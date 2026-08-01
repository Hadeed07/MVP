from fastapi import APIRouter, UploadFile, HTTPException, Response
import numpy as np
import cv2

from schemas import ScanResponse, SpineResult
from services.pipeline import SpinePipeline
from services.text_cleaning import clean_ocr
from store import save_image, get_image

router = APIRouter()

# Initialize the pipeline once when the application starts.
pl = SpinePipeline()


@router.post("/scan", response_model=ScanResponse)
def scan(file: UploadFile):
    """
    Upload a bookshelf image and return detected book spines
    together with a scan ID for later image retrieval.
    """
    
    try:
        contents = file.file.read()
    finally:
        file.file.close()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    matches, obb_corners = pl.results(img)
    print("SCAN.PY UPDATED")

    print("OCR RESULTS TYPE:", type(matches))
    print("FIRST ITEM:", matches[0] if matches else "EMPTY")

    spines = []
    for book in matches:
        idx = book["spine_idx"]
        text = book["ocr_text"]

        if text:
            corners = pl.normalize(pl.order_points(obb_corners[idx]),img)
            spines.append(SpineResult(id=str(idx),text=text,corners=corners))


    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")

    scan_id = save_image(buffer.tobytes())

    return ScanResponse(scan_id=scan_id, spines=spines)


@router.get("/scan/{scan_id}/image")
def get_scan_image(scan_id: str):
    """
    Retrieve the original uploaded image using its scan ID.
    """
    
    image_bytes = get_image(scan_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="scan_id not found")
    return Response(content=image_bytes, media_type="image/jpeg")