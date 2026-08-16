"""YOLO OBB detection, annotation and perspective crop extraction."""

import cv2
from typing import Any
import numpy as np
from .config import YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD
from .utils import order_points


def detect_spines(
    model: Any,
    image: np.ndarray,
    confidence_threshold: float=YOLO_CONFIDENCE_THRESHOLD,
    iou_threshold: float=YOLO_IOU_THRESHOLD,
) -> np.ndarray:
    """Run YOLO OBB detection."""
    results = model.predict(
        image,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    return results[0].obb.xyxyxyxy.cpu().numpy()


def annotate_detections(image: np.ndarray, obb_corners: np.ndarray) -> np.ndarray:
    """Draw each YOLO OBB and its crop index."""
    annotated = image.copy()

    for crop_idx, corners in enumerate(obb_corners):
        points = np.asarray(
            corners,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            annotated,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        centroid = points.reshape(-1, 2).mean(axis=0)
        cx = int(centroid[0])
        cy = int(centroid[1])

        label = str(crop_idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        text_x = cx - tw // 2
        text_y = cy + th // 2

        cv2.rectangle(
            annotated,
            (
                text_x - 4,
                text_y - th - baseline - 2,
            ),
            (
                text_x + tw + 4,
                text_y + baseline + 2,
            ),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def crop_spines(image: np.ndarray, obb_corners: np.ndarray) -> list[np.ndarray | None]:
    """Perspective-correct each detected spine and preserve crop order."""
    crops = []

    for corners in obb_corners:
        corners = order_points(corners)

        width = int(
            max(
                np.linalg.norm(corners[2] - corners[3]),
                np.linalg.norm(corners[1] - corners[0]),
            )
        )

        height = int(
            max(
                np.linalg.norm(corners[1] - corners[2]),
                np.linalg.norm(corners[0] - corners[3]),
            )
        )

        if width < 2 or height < 2:
            crops.append(None)
            continue

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            corners,
            destination,
        )

        crop = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(
                crop,
                cv2.ROTATE_90_CLOCKWISE,
            )

        crops.append(crop)

    return crops
