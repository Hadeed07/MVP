from fastapi import APIRouter, UploadFile, HTTPException, Response
import numpy as np
import cv2
import os
from dotenv import load_dotenv

from schemas import ScanResponse, RecommendedBook
from pipeline_modules.pipeline import SpinePipeline
from services.text_cleaning import clean_ocr
from store import save_image, get_image

router = APIRouter()

load_dotenv()
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API")

# Initialize the pipeline once when the application starts.
pl = SpinePipeline(
    catalog_path=r"Dataset\Books.csv",
    chroma_path=r"chroma_db",
    collection_name="books",
    google_books_api_key=GOOGLE_BOOKS_API_KEY,
)


@router.post("/scan", response_model=ScanResponse)
def scan(file: UploadFile, query: str | None = None):
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
        results = pl.results(img, query=query)

        print("D - Pipeline finished")

        # ---------------------------------------------------------
        # 1. Get detected spine geometry
        # ---------------------------------------------------------

        crop_records = results['crop_records']

        # Map crop index to polygon corners
        polygon_by_crop = {record['crop_idx']: record['bbox'] for record in crop_records}
        print(f"Detected {len(polygon_by_crop)} spine polygons")

        # ---------------------------------------------------------
        # 2. Get query recommendations
        # ---------------------------------------------------------

        recommendations_df = results["top_query_recommendations"]
        recommended_books = []

        if recommendations_df is not None and not recommendations_df.empty:
            for _, book in recommendations_df.iterrows():
                crop_idx = book.get("Crop Index")
                if crop_idx is None:
                    continue

                # Pandas may return numpy integer types.
                crop_idx = int(crop_idx)
                corners = polygon_by_crop.get(crop_idx)

                if corners is None:
                    print(f"WARNING: No polygon found for crop_idx={crop_idx}")
                    continue

                recommended_books.append(
                    RecommendedBook(
                        id=str(crop_idx),
                        crop_idx=crop_idx,
                        title=str(book.get("Title", "")),
                        author=str(book.get("Author", "")),
                        isbn13=str(book.get("ISBN13", "")),
                        description=str(book.get("Description", "")),
                        thumbnail=str(book.get("Thumbnail", "")),
                        query_score=float(book.get("Query Score", 0.0)),
                        corners=corners,
                    )
                )

        print(f"E - Built {len(recommended_books)} recommended books")


        # ---------------------------------------------------------
        # 3. Save the original image
        # ---------------------------------------------------------

        success, buffer = cv2.imencode(".jpg", img)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image",)

        print("F - Image encoded")
        scan_id = save_image(buffer.tobytes())

        print("G - Image saved")
        print("scan_id:", scan_id)

        # ---------------------------------------------------------
        # 4. Build API response
        # ---------------------------------------------------------

        response = ScanResponse(
            scan_id=scan_id,
            image_width=img.shape[1],
            image_height=img.shape[0],
            recommendations=recommended_books,
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