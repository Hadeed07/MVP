"""Display helpers for the notebook pipeline."""

from IPython.display import display
from PIL import Image
import cv2


def display_results(output):
    """Display the same major output sections as the working pipeline."""
    if output.get("yolo_output_image") is not None:
        print("YOLO DETECTIONS (CROP NUMBERS)")
        print("-" * 80)

        display(
            Image.fromarray(
                cv2.cvtColor(
                    output["yolo_output_image"],
                    cv2.COLOR_BGR2RGB,
                )
            )
        )

    print("=" * 80)
    print("BOOK DETECTION SUMMARY")
    print("=" * 80)
    print(f"Detected Spines : {output['num_detected_spines']}")
    print(f"OCR Results     : {output['num_ocr_results']}")
    print(f"Matched Books   : {output['num_matched_books']}")
    print(f"Unmatched Books : {output['num_unmatched_books']}")

    print("\nPER-CROP PIPELINE PROGRESS")
    print("-" * 80)
    display(output["crop_tracking_df"])

    print("\nMATCHED BOOKS")
    print("-" * 80)
    display(output["matched_books"])

    if not output["top_query_recommendations"].empty:
        print("\nTOP QUERY MATCHES")
        print("-" * 80)
        display(output["top_query_recommendations"])

    if output["unmatched_queries"]:
        print("\nUNMATCHED OCR QUERIES")
        print("-" * 80)

        for ocr_query in output["unmatched_queries"]:
            print(f"• {ocr_query}")

    if output.get("timing_df") is not None:
        print("\nPIPELINE TIMING")
        print("-" * 80)
        display(output["timing_df"])
