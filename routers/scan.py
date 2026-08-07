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

    print("A - Endpoint entered")

    try:
        try:
            contents = file.file.read()
        finally:
            file.file.close()

        print("B - File read")

        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        print("C - Image decoded")

        matches, obb_corners = pl.results(img)

        print("D - Pipeline finished")
        print("SCAN.PY UPDATED")

        print("OCR RESULTS TYPE:", type(matches))
        print("FIRST ITEM:", matches[0] if matches else "EMPTY")

        spines = []

        for book in matches:
            idx = book["spine_idx"]
            text = book["ocr_text"]

            if text:
                corners = pl.normalize(
                    pl.order_points(obb_corners[idx]),
                    img
                )

                spines.append(
                    SpineResult(
                        id=str(idx),
                        text=text,
                        corners=corners,
                    )
                )

        print(f"E - Built {len(spines)} spines")

        success, buffer = cv2.imencode(".jpg", img)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to encode annotated image",
            )

        print("F - Image encoded")

        scan_id = save_image(buffer.tobytes())

        print("G - Image saved")
        print("scan_id:", scan_id)

        response = ScanResponse(
            scan_id=scan_id,
            spines=spines,
        )

        print("H - Response object created")
        print("Returning response...")

        return response

    except Exception as e:
        import traceback

        print("========== EXCEPTION ==========")
        traceback.print_exc()
        print("===============================")
        raise

    finally:
        print("I - Finally block executed")


@router.get("/scan/{scan_id}/image")
def get_scan_image(scan_id: str):
    """
    Retrieve the original uploaded image using its scan ID.
    """
    
    image_bytes = get_image(scan_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="scan_id not found")
    return Response(content=image_bytes, media_type="image/jpeg")