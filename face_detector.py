from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(
        self, tm_window_size: int = 20, tm_threshold: float = 0.7, aligned_image_size: int = 224 #tried 30,0.5
    ) -> None:
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference: Optional[FaceDetectionResult] = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
         # tm_window_size: search-window margin (in pixels) around the last known bounding box,
        # i.e., we search within (bbox expanded by ±tm_window_size each side).
        self.tm_window_size = int(tm_window_size)

        # Similarity threshold for declaring tracking failure and triggering re-initialization.
        # For TM_CCOEFF_NORMED higher is better (range approx. [-1, 1]).
        self.tm_threshold = float(tm_threshold)

        # Matching mode: normalized correlation coefficient works well in practice for faces.
        self.tm_method = cv2.TM_CCOEFF_NORMED

    def _clip_rect(
        self, rect: tuple[int, int, int, int], image_shape: tuple[int, int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        """Clip a (x,y,w,h) rectangle to image bounds; return None if invalid."""
        x, y, w, h = rect
        H, W = image_shape[0], image_shape[1]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Ensure positive width/height first.
        if w <= 0 or h <= 0:
            return None

        # Clip top-left.
        x_clipped = max(x, 0)
        y_clipped = max(y, 0)

        # Reduce width/height if rect goes out of bounds.
        w_clipped = min(w - (x_clipped - x), W - x_clipped)
        h_clipped = min(h - (y_clipped - y), H - y_clipped)

        if w_clipped <= 1 or h_clipped <= 1:
            return None

        return (x_clipped, y_clipped, int(w_clipped), int(h_clipped))



    # TODO: Track a face in a new image using template matching.
    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:

        # If no reference exists (first frame or previous failure), initialize using MTCNN.
        if self.reference is None:
            det = self.detect_face(image)
            self.reference = det
            return det

        # Sanitize reference rect (MTCNN can return boxes that touch/overflow image borders).
        ref_rect = self._clip_rect(self.reference.rect, self.reference.image.shape)
        if ref_rect is None:
            det = self.detect_face(image)
            self.reference = det
            return det

        # Build template from reference frame.
        template_bgr = self.crop_face(self.reference.image, ref_rect)
        if template_bgr.size == 0:
            det = self.detect_face(image)
            self.reference = det
            return det

        # Convert to grayscale for template matching (robust and faster).
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        x, y, w, h = ref_rect

        # Define a search window around the last known position: bbox expanded by ±tm_window_size.
        margin = self.tm_window_size
        H, W = image.shape[0], image.shape[1]
        sx0 = max(x - margin, 0)
        sy0 = max(y - margin, 0)
        sx1 = min(x + w + margin, W)
        sy1 = min(y + h + margin, H)

        search_bgr = image[sy0:sy1, sx0:sx1, :]
        if search_bgr.size == 0:
            det = self.detect_face(image)
            self.reference = det
            return det

        search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # If the search window is smaller than the template, re-initialize.
        if (
            search_gray.shape[0] < template_gray.shape[0]
            or search_gray.shape[1] < template_gray.shape[1]
        ):
            det = self.detect_face(image)
            self.reference = det
            return det

        # Template matching.
        response = cv2.matchTemplate(search_gray, template_gray, self.tm_method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)

        # Interpret match score depending on method.
        if self.tm_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            score = float(min_val)  # lower is better
            best_loc = min_loc
            tracking_ok = score <= self.tm_threshold
        else:
            score = float(max_val)  # higher is better
            best_loc = max_loc
            tracking_ok = score >= self.tm_threshold

        # If tracking fails (e.g., large pose change/occlusion), re-initialize with MTCNN.
        if not tracking_ok:
            det = self.detect_face(image)
            self.reference = det
            return det

        # Compute new bbox position in full image coordinates.
        new_x = int(sx0 + best_loc[0])
        new_y = int(sy0 + best_loc[1])
        new_rect = (new_x, new_y, w, h)

        # Clip bbox to image bounds (important near borders).
        clipped_new_rect = self._clip_rect(new_rect, image.shape)
        if clipped_new_rect is None:
            det = self.detect_face(image)
            self.reference = det
            return det

        # Alignment integrated into tracking (pose/size normalization to 224x224).
        aligned = self.align_face(image, clipped_new_rect)
        result = FaceDetectionResult(rect=clipped_new_rect, image=image, aligned=aligned)

        # Update reference to adapt to gradual appearance changes and keep the search local.
        self.reference = result
        return result
        # return None

    

    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        # MTCNN typically expects RGB images.
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            self.reference = None
            return None

        # If the frame is extremely small, MTCNN can fail internally.
        if image.shape[0] < 40 or image.shape[1] < 40:
            self.reference = None
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            detections = self.detector.detect_faces(
                rgb, threshold_pnet=0.85, threshold_rnet=0.9
            )
        except ValueError:
            # Known failure mode in some mtcnn/keras combos when no proposals exist.
            self.reference = None
            return None
        except Exception:
            # Defensive: treat any unexpected MTCNN runtime error as "no detection".
            self.reference = None
            return None

        if not detections:
            self.reference = None
            return None

        # Select face with the largest bounding box.
        largest_detection = int(np.argmax([d["box"][2] * d["box"][3] for d in detections]))
        face_rect = detections[largest_detection]["box"]

        # Align the detected face (alignment operates on original BGR image).
        aligned = self.align_face(image, face_rect)
        return FaceDetectionResult(rect=face_rect, image=image, aligned=aligned)

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
