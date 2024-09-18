import math
import os
import cv2
import numpy as np
import yolov5
from PIL import Image
from typing import Optional
from deepface import DeepFace

from app_utils.logging import get_logger
from app_utils.util import class_Order, four_point_transform, get_center_point, order_points
import config as cfg

logger = get_logger(__name__)


class Detector:
    """
    Detector class for detecting and processing Vietnamese ID cards (CCCD)
    and faces using YOLO models and a QR code reader.
    """

    def __init__(self) -> None:
        """
        Initializes the Detector object by loading the required YOLO models
        for corner, face detection, and QR code reading.
        """
        # Load YOLO models once during initialization to avoid repeated loading
        self.CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
        self.CORNER_MODEL2 = yolov5.load(cfg.CORNER_MODEL_PATH2)
        self.FACE_MODEL = yolov5.load(cfg.FACE_MODEL_PATH)

        self.CORNER_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
        self.CORNER_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD
        self.CORNER_MODEL2.conf = cfg.CONF_CONTENT_THRESHOLD
        self.CORNER_MODEL2.iou = cfg.IOU_CONTENT_THRESHOLD

    def detect_cccd(self, img_path: str, is_cccd_back: bool) -> Optional[np.ndarray]:
        """
        Detects the ID card (CCCD) from the given image and corrects orientation based on face detection or corner detection.

        Args:
            img_path (str): The path to the image file.
            is_cccd_back (bool): A flag indicating if the image is of the back side of the ID card.

        Returns:
            Optional[np.ndarray]: The aligned ID card image or a cropped portion if corners are not detected.
                                Returns None if detection fails.
        """
        # Attempt to load the image
        if not os.path.exists(img_path):
            logger.error(f"Image not found: {img_path}")
            return None

        img = cv2.imread(img_path)
        if img is None:  # OpenCV failed
            try:
                pil_image = Image.open(img_path)
                img = cv2.cvtColor(
                    np.array(pil_image), cv2.COLOR_RGB2BGR
                )  # Convert to OpenCV format
            except Exception as e:
                logger.error(f"Failed to open image with PIL: {e}")
                return None

        if is_cccd_back:
            # Process for the back side of the card
            return self.__detect_and_align_back(img)

        else:
            # For the front side, use face detection to correct orientation
            return self.__detect_and_align_front(img, img_path)

    def __detect_and_align_back(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects and aligns the back of the CCCD card using corner detection models.

        Args:
            img (np.ndarray): The image of the back side of the CCCD card.

        Returns:
            Optional[np.ndarray]: The aligned ID card image or a cropped portion if corners are not detected.
                                  Returns None if detection fails.
        """
        predictions = self.CORNER_MODEL(img).pred[0]
        categories = predictions[:, 5].tolist()

        if len(categories) == 4:
            boxes = class_Order(predictions[:, :4].tolist(), categories)
            corners = [get_center_point(box) for box in boxes]

            # Auto-correct the card using the detected corners
            aligned = self.auto_correct_card(img, corners)
            return aligned

        predictions2 = self.CORNER_MODEL2(img).pandas().xyxy[0]
        if predictions2.empty:
            raise ValueError("Missing fields! Detecting CCCD failed!")

        # Extract bounding box coordinates
        coords = predictions2[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)[0]
        result = img[coords[1] : coords[3], coords[0] : coords[2]]

        return result

    def __detect_and_align_front(
        self, img: np.ndarray, img_path: str
    ) -> Optional[np.ndarray]:
        """
        Detects and aligns the front of the CCCD card using face detection and corner detection.

        Args:
            img (np.ndarray): The image of the front side of the CCCD card.
            img_path (str): The path to the image file.

        Returns:
            Optional[np.ndarray]: The aligned ID card image or a cropped portion if corners are not detected.
                                  Returns None if detection fails.
        """
        # Use face detection to correct orientation
        aligned_image = self.__detect_face_fixed(img_path)
        if aligned_image is not None:
            img = np.array(aligned_image)
        else:
            logger.error("Face detection failed! Skipping face alignment.")
            return None

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Run the first corner detection model
        predictions = self.CORNER_MODEL(img).pred[0]
        categories = predictions[:, 5].tolist()

        if len(categories) == 4:  # Process corners if detected
            boxes = class_Order(predictions[:, :4].tolist(), categories)
            center_points = np.array([get_center_point(box) for box in boxes])
            center_points[2:, 1] += 30  # Adjust points by 30 pixels

            aligned = four_point_transform(pil_img, center_points)
            return np.array(aligned)

        # Fallback to the second corner detection model if corners are missing
        predictions2 = self.CORNER_MODEL2(img).pandas().xyxy[0]
        if predictions2.empty:
            raise ValueError("Missing fields! Detecting CCCD failed!")

        # Extract bounding box coordinates
        coords = predictions2[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)[0]
        result = img[coords[1] : coords[3], coords[0] : coords[2]]

        return result
    
    def visualize_detection(self,img, facial_area):
        """
        Visualize the face bounding box and eyes on the image.
        
        Args:
            img (np.ndarray): The original image.
            facial_area (dict): The detected face bounding box and eye positions.
        """
        # Draw the face bounding box
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the eyes
        left_eye = facial_area["left_eye"]
        right_eye = facial_area["right_eye"]
        cv2.circle(img, left_eye, 5, (0, 0, 255), -1)
        cv2.circle(img, right_eye, 5, (0, 0, 255), -1)

        cv2.imshow("Face Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def __correct_rotation(self, img: np.ndarray, facial_area: dict) -> np.ndarray:
        """
        Corrects the rotation of the image based on the detected face coordinates.

        Args:
            img (np.ndarray): The original image as a NumPy array.
            facial_area (dict): The bounding box coordinates of the detected face with 'left_eye' and 'right_eye'.

        Returns:
            np.ndarray: The rotated image with corrected orientation.
        """

        # Calculate the rotation angle based on facial features
        face_angle = self.__calculate_face_angle(facial_area)

        # Normalize the angle to the range [-180, 180]
        if face_angle > 180:
            face_angle -= 360
        elif face_angle < -180:
            face_angle += 360

        # Determine the rotation direction and apply the correction
        if abs(face_angle) > 90:
            angle_card = face_angle + 180 if face_angle < 0 else face_angle - 180
        else:
            angle_card = -face_angle

        # Apply the rotation
        height, width = img.shape[:2]  # Image dimensions
        image_center = (width / 2, height / 2)  # Center of rotation
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle_card, 1.0)

        # Calculate new bounds
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # Adjust the rotation matrix to take the new bounds into account
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # Rotate the image with the new bounds
        rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

        return rotated_img


    def auto_correct_card(self, image_path: str, corners: list) -> np.ndarray:
        """
        Corrects the perspective of an ID card based on the detected corners.

        Args:
            image_path (str): The path to the image file.
            corners (list): A list of four corner points detected in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

        Returns:
            np.ndarray: The corrected ID card image.
        """
        # Step 1: Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read the image from path: {image_path}")

        # Step 2: Order the corner points in the correct order
        corners = np.array(corners, dtype="float32")
        ordered_corners = order_points(corners)

        # Step 3: Define the target rectangle size for the ID card
        width = 856  # Standard width of the ID card
        height = 540  # Standard height of the ID card

        # Define destination points for perspective transformation
        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        # Step 4: Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)

        # Step 5: Apply the perspective transform to the image
        warped = cv2.warpPerspective(image, M, (width, height))

        # Return the corrected image
        return warped

    # def _GetInformationAndSave(self, _results, _idnumber, _idnumberbox):
    #     """
    #     Extracts information from OCR results and saves them to a JSON file.

    #     Args:
    #         _results (list): The OCR results as a list of tuples (text, bounding_box).
    #         _idnumber (str): The ID number extracted.
    #         _idnumberbox (list): The bounding box coordinates for the ID number.
    #     """
    #     logger.info("Extracting information from OCR results...")
    #     result = {
    #         "ID_number": _idnumber,
    #         "Name": "",
    #         "Date_of_birth": "",
    #         "Gender": "",
    #         "Nationality": "",
    #         "Place_of_origin": "",
    #         "Place_of_residence": "",
    #         "ID_number_box": _idnumberbox,
    #     }
    #     regex_dob = r"[0-9][0-9]/[0-9][0-9]"
    #     regex_residence = r"[0-9][0-9]/[0-9][0-9]/|[0-9]{4,10}|Date|Demo|Dis|Dec|Dale|fer|ting|gical|ping|exp|ver|pate|cond|trị|đến|không|Không|Có|Pat|ter|ity"
    #     for i, res in enumerate(_results):
    #         s = res[0]
    #         if re.search(r"tên|name", s):
    #             Name = (_results[i + 1]if (not re.search(r"[0-9]", _results[i + 1][0]))else _results[i + 2])
    #             result["Name"] = Name[0].title()
    #             result["Name_box"] = Name[1] if Name[1] else []
    #             if result["Date_of_birth"] == "":
    #                 DOB = (_results[i - 2]if re.search(regex_dob, _results[i - 2][0])else [])
    #                 result["Date_of_birth"] = ((re.split(r":|\s+", DOB[0]))[-1].strip() if DOB else "")
    #                 result["Date_of_birth_box"] = DOB[1] if DOB else []
    #             continue
    #         if re.search(r"sinh|birth|bith", s) and (not result["Date_of_birth"]):
    #             if re.search(regex_dob, s):
    #                 DOB = _results[i]
    #             elif re.search(regex_dob, _results[i - 1][0]):
    #                 DOB = _results[i - 1]
    #             elif re.search(regex_dob, _results[i + 1][0]):
    #                 DOB = _results[i + 1]
    #             else:
    #                 DOB = []
    #             result["Date_of_birth"] = ((re.split(r":|\s+", DOB[0]))[-1].strip() if DOB else "")
    #             result["Date_of_birth_box"] = DOB[1] if DOB else []
    #             if re.search(r"Việt Nam", _results[i + 1][0]):
    #                 result["Nationality"] = "Việt Nam"
    #                 result["Nationality_box"] = _results[i + 1][1]
    #             continue
    #         if re.search(r"Giới|Sex", s):
    #             Gender = _results[i]
    #             result["Gender"] = "Nữ" if re.search(r"Nữ|nữ", Gender[0]) else "Nam"
    #             result["Gender_box"] = Gender[1] if Gender[1] else []
    #             continue
    #         if re.search(r"Quốc|tịch|Nat", s):
    #             if (not re.search(r"ty|ing", re.split(r":|,|[.]|ty|tịch", s)[-1].strip())and len(re.split(r":|,|[.]|ty|tịch", s)[-1].strip()) >= 3):
    #                 Nationality = _results[i]
    #             elif not re.search(r"[0-9][0-9]/[0-9][0-9]/", _results[i + 1][0]):
    #                 Nationality = _results[i + 1]
    #             else:
    #                 Nationality = _results[i - 1]
    #             result["Nationality"] = (re.split(r":|-|,|[.]|ty|[0-9]|tịch", Nationality[0])[-1].strip().title())
    #             result["Nationality_box"] = Nationality[1] if Nationality[1] else []
    #             for s in re.split(r"\s+", result["Nationality"]):
    #                 if len(s) < 3:
    #                     result["Nationality"] = (re.split(s, result["Nationality"])[-1].strip().title())
    #             if re.search(r"Nam", result["Nationality"]):
    #                 result["Nationality"] = "Việt Nam"
    #             continue
    #         if re.search(r"Quê|origin|ongin|ngin|orging", s):
    #             PlaceOfOrigin = ([_results[i], _results[i + 1]]if not re.search(r"[0-9]{4}", _results[i + 1][0])else [])
    #             if PlaceOfOrigin:
    #                 if (len(re.split(r":|;|of|ging|gin|ggong", PlaceOfOrigin[0][0])[-1].strip())> 2):
    #                     result["Place_of_origin"] = ((re.split(r":|;|of|ging|gin|ggong", PlaceOfOrigin[0][0]))[-1].strip()+ ", "+ PlaceOfOrigin[1][0])
    #                 else:
    #                     result["Place_of_origin"] = PlaceOfOrigin[1][0]
    #                 result["Place_of_origin_box"] = PlaceOfOrigin[1][1]
    #             continue
    #         if re.search(r"Nơi|trú|residence", s):
    #             vals2 = (
    #                 ""
    #                 if (i + 2 > len(_results) - 1)
    #                 else (
    #                     _results[i + 2] if len(_results[i + 2][0]) > 5 else _results[-1]
    #                 )
    #             )
    #             vals3 = (""if (i + 3 > len(_results) - 1)else (_results[i + 3] if len(_results[i + 3][0]) > 5 else _results[-1]))
    #             if (re.split(r":|;|residence|ence|end", s))[-1].strip() != "":
    #                 if vals2 != "" and not re.search(regex_residence, vals2[0]):
    #                     PlaceOfResidence = [_results[i], vals2]
    #                 elif vals3 != "" and not re.search(regex_residence, vals3[0]):
    #                     PlaceOfResidence = [_results[i], vals3]
    #                 elif not re.search(regex_residence, _results[-1][0]):
    #                     PlaceOfResidence = [_results[i], _results[-1]]
    #                 else:
    #                     PlaceOfResidence = [_results[-1], []]
    #             else:
    #                 PlaceOfResidence = (
    #                     [vals2, []]
    #                     if (vals2 and not re.search(regex_residence, vals2[0]))
    #                     else [_results[-1], []]
    #                 )

    #             logger.info("PlaceOfResidence: {}".format(PlaceOfResidence))
    #             if PlaceOfResidence[1]:
    #                 result["Place_of_residence"] = (
    #                     re.split(
    #                         r":|;|residence|sidencs|ence|end", PlaceOfResidence[0][0]
    #                     )[-1].strip()
    #                     + " "
    #                     + str(PlaceOfResidence[1][0]).strip()
    #                 )
    #                 result["Place_of_residence_box"] = PlaceOfResidence[1][1]
    #             else:
    #                 result["Place_of_residence"] = PlaceOfResidence[0][0]
    #                 result["Place_of_residence_box"] = (
    #                     PlaceOfResidence[0][1] if PlaceOfResidence else []
    #                 )
    #             continue
    #     return result
