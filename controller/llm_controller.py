
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import base64
from app_utils.file_handler import convert_image
from app_utils.logging import get_logger
from config import URL_API_LLM
from ollama import AsyncClient

logger = get_logger(__name__)

@dataclass
class QRCodeData:
    id_number: str = ""
    id_number_old: str = ""
    fullname: str = ""
    day_of_birth: str = ""
    sex: str = ""
    place_of_residence: str = ""
    date_of_issue: str = ""

    @classmethod
    def from_qr_string(cls, qr_data: str) -> 'QRCodeData':
        parts = qr_data.strip().split("|")
        return cls(
            id_number=parts[0] if len(parts) > 0 else "",
            id_number_old=parts[1] if len(parts) > 1 else "",
            fullname=parts[2] if len(parts) > 2 else "",
            day_of_birth=parts[3] if len(parts) > 3 else "",
            sex=parts[4] if len(parts) > 4 else "",
            place_of_residence=parts[5] if len(parts) > 5 else "",
            date_of_issue=parts[6] if len(parts) > 6 else ""
        )

class LlmController:
    DEFAULT_MODEL = "qwen2.5:14b"
    PLACE_OF_ISSUE_DIRECTOR = "Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội"
    PLACE_OF_ISSUE_DEFAULT = "Bộ Công An"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.client = AsyncClient(host=URL_API_LLM)
        self.model = model
        self.system_prompt = self._get_default_system_prompt()
        self.user_context = None

    def _get_default_system_prompt(self) -> str:
        return """You are an AI assistant specialized in processing OCR text from Vietnamese Citizen ID cards (CCCD).
Your task is to analyze, compare, and correct OCR errors while cross-referencing with QR code data to ensure accuracy.

Please follow these steps:
1. Compare OCR sources with QR code data, prioritizing QR code data for discrepancies
2. Correct OCR errors (spelling, diacritics, formatting)
3. Format dates as 'DD/MM/YYYY'
4. Leave blank ('') if information is unclear/missing
5. Only modify existing data; no new information
6. Return JSON format only, no explanations
7. No text except JSON response"""

    def _normalize_qr_data(self, qr_data: Any) -> str:
        """Normalize QR code data to string format."""
        if isinstance(qr_data, (list, tuple)):
            return qr_data[0] if qr_data else ""
        return str(qr_data).strip()

    def _get_ocr_result(self, ocr_results: Dict[str, Any], key: str, subkey: str) -> str:
        """Safely extract OCR results."""
        return ocr_results.get(key, {}).get(subkey, "")

    def _generate_user_context(self, ocr_results: Dict[str, Any]) -> str:
        try:
            # Extract OCR results
            front_ocr = self._get_ocr_result(ocr_results, "front_side_ocr", "image1_result")
            back_ocr = self._get_ocr_result(ocr_results, "front_side_ocr", "image2_result")

            # Process QR code data
            front_qr = self._normalize_qr_data(ocr_results.get("front_side_qr", ""))
            back_qr = self._normalize_qr_data(ocr_results.get("back_side_qr", ""))
            qr_data = front_qr or back_qr

            # Parse QR code data
            qr_info = QRCodeData.from_qr_string(qr_data)

            return self._format_context(front_ocr, back_ocr, qr_info)

        except Exception as e:
            logger.error(f"Error generating user context: {str(e)}")
            raise ValueError("Failed to generate user context") from e

    def _format_context(self, front_ocr: str, back_ocr: str, qr_info: QRCodeData) -> str:
        return f"""Data extracted from OCR data and QR data of the Vietnamese Citizen ID Card (CCCD):

### Data from front side ID Card OCR:
{front_ocr}

### Data from back side ID Card OCR:
{back_ocr}

QR code data:
- id_number: {qr_info.id_number}
- id_number_old: {qr_info.id_number_old}
- fullname: {qr_info.fullname}
- day_of_birth: {qr_info.day_of_birth}
- sex: {qr_info.sex}
- place_of_residence: {qr_info.place_of_residence}
- date_of_issue: {qr_info.date_of_issue}

"Please analyze and edit the information based on the following instructions:\n"
"1. Compare data from OCR, OCR LLM  and QR code ; prioritize QR code data in case of discrepancies. If some information is missing in the QR code, use data from OCR.\n"
"3. Ensure all dates follow the DD/MM/YYYY format.\n"
"4. Leave blank ('') if information is completely unreadable. If possible, infer information from OCR or QR code.\n"
"5. Ensure that the value for 'sex' is written out fully (e.g., 'Nam' instead of 'N').\n"
"6. Fix common OCR errors, such as misidentified characters (e.g., 'Giói tính' to 'Giới tính', 'Ngày sinh/Ide date of birth' to 'Ngày sinh/Date of birth').\n"
"7. For phrases like 'Có giá trị đến: [date]' or 'Date of expiry', recognize that this refers to 'date_of_expiration' and separate it from 'place_of_residence'. 'Date of expiry' may appear on either the front or back of the CCCD.\n"
"8. If there is one date on the back of the CCCD, it is the 'date_of_issue'. If there are two dates, they are 'date_of_issue' and 'date_of_expiration'; the issue date will precede the expiration date.\n"
"9. When extracting 'place_of_origin' and 'place_of_residence', ensure they are ordered from the smallest to largest administrative unit.\n"
"10. Ensure 'nationality' is spelled correctly and with full Vietnamese accents for any country (e.g., 'Việt Nam' instead of 'Vietnam').\n"
"11. For 'place_of_issue', extract the value from the OCR data as follows:\n"
"    - If 'Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội | QUẢN LÝ HÀNH CHÍNH VÈ TRẬT TỰ XÃ HỘI | DIRECTOR GENERAL OF THE POLICE DEPARTMENT FOR ADMINISTRATIVE MANAGEMENT OF SOCIAL ORDER' appears or it have part in the OCR data, use 'Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội' as 'place_of_issue'.\n"
"    - If not, use 'Bộ Công An'.\n"
"12. Combine information from all sources to fill out the following 11 fields completely.\n"
"13. Prioritize accuracy, spelling corrections, and formatting.\n"
"14. Ensure all dates are in DD/MM/YYYY format.\n"
"15. Be mindful of unwanted lines; combine or separate information based on context.\n"
"16. If information is not available in the QR code, use data from OCR.\n"
"17. Correct misidentifications of personal names, places, and technical terms.\n"
"18. Ensure the value for 'sex' is either 'Nam' or 'Nữ'.\n"
"19. If the address for 'place_of_residence' or 'place_of_origin' is split across multiple lines in the OCR data, recognize that these lines belong together and combine them into a single, complete address.\n"
"20. Do not change personal names unless you are certain there is a misspelling. If unsure, keep the personal names exactly as they appear in the OCR data.\n\n"
"21. Return the result in Vietnamese and ensure it contains complete information (full path information from OCR ) for both 'nơi thường trú' (place_of_residence) and 'quê quán | Nơi Sinh' (place_of_origin).\n"
"22. Only export information when available, do not add or remove information"
"Return the replie as JSON  not explain with the following 11 fields:\n"
Return JSON with these fields:
{{
    "id_number": "",
    "id_number_old": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "",
    "nationality": "",
    "place_of_residence": "",
    "place_of_origin": "",
    "date_of_expiration": "",
    "date_of_issue": "",
    "place_of_issue": ""
}}"""

    async def _send_message(
        self, 
        system_prompt: str, 
        user_context: str, 
        image: Optional[Any] = None
    ) -> Dict[str, Any]:
        if not all([system_prompt, user_context]):
            raise ValueError("System prompt and user context must be set before sending messages.")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context}
        ]

        if image:
            try:
                image_bytes = convert_image(image)
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                messages.append({"role": "user", "content": image_base64})
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise ValueError("Failed to process image") from e

        return await self.client.chat(self.model, messages=messages)

    # Public methods remain largely the same but with improved error handling
    def set_model(self, model: str) -> None:
        """Set the model to be used for inference."""
        self.model = model

    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt."""
        self.system_prompt = prompt

    def set_user_context(self, ocr_results: Dict[str, Any]) -> str:
        """Set user context from OCR results."""
        self.user_context = self._generate_user_context(ocr_results)
        return self.user_context

    async def send_message(self, image: Optional[Any] = None) -> Dict[str, Any]:
        """Send a message using default prompt and context."""
        return await self._send_message(self.system_prompt, self.user_context, image)

    async def send_custom_message(
        self,
        custom_prompt: Optional[str] = None,
        custom_context: Optional[str] = None,
        custom_image: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Send a message with custom prompt and/or context."""
        return await self._send_message(
            custom_prompt or self.system_prompt,
            custom_context or self.user_context,
            custom_image
        )