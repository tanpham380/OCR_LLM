import base64
from typing import Any, Dict, Optional, Tuple
from app_utils.file_handler import convert_image
from app_utils.logging import get_logger
from config import URL_API_LLM
from ollama import AsyncClient

logger = get_logger(__name__)


class LlmController:
    def __init__(self, model: str = "finalend/llama-3.1-storm:8b") -> None:
        self.client = AsyncClient(host=URL_API_LLM)
        self.model = model
        self.system_prompt = self._get_default_system_prompt()
        self.user_context = None

    def _get_default_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in processing OCR text from Vietnamese Citizen Identity Cards (CCCD). Your task is to analyze, compare, and correct OCR errors from various sources such as VietOCR, EasyOCR, and PaddleOCR, while cross-referencing with QR code data to ensure accuracy. Please follow these steps:"
            " 1. Compare information from OCR sources with QR code data. Prioritize QR code data if there are discrepancies."
            " 2. If the QR code is empty or missing information, use data from OCR."
            " 3. Correct OCR errors including spelling mistakes, missing diacritics, or incorrect formatting."
            " 4. Ensure all dates are formatted as 'DD/MM/YYYY'."
            " 5. Leave fields empty ('') if the information is unclear or missing."
            " 6. Only edit existing data; do not add new information."
            " 7. Handle special cases:"
            "    - Look for 'place_of_origin' and 'place_of_residence' on both front and back OCR."
            "    - Identify 'expiration_date' from phrases like 'Có giá trị đến:', 'Date of expiry', 'Ngày, tháng, năm hết hạn'."
            "    - Process information that spans multiple lines or is split across different parts."
            " 8. The result must be in JSON format only."
            " 9. Include all fields in the JSON, even if empty."
            " 10. DO NOT include any explanations, comments, or additional text outside the JSON structure."
        )


    def set_model(self, model: str) -> None:
        self.model = model

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def set_user_context(self, ocr_results: Dict[str, Any]) -> str:
        """
        Sets the user context by generating the combined context from both front and back OCR and QR data,
        with a fallback to the back QR code if the front QR code is missing.

        Args:
            ocr_results (Dict[str, Any]): Combined OCR results from the front and back images.
        """
        self.user_context = self._generate_user_context(ocr_results)
        return self.user_context

    def _generate_user_context(self, ocr_results: Dict[str, Any]) -> str:
        try:
            front_ocr = (
                ocr_results.get("front_side_ocr", {})
                .get("package_ocr", "")
                .replace("\\", "")
            )
            back_ocr = (
                ocr_results.get("back_side_ocr", {})
                .get("package_ocr", "")
                .replace("\\", "")
            )

            front_qr_code_data = ocr_results.get("front_side_qr", "")
            back_qr_code_data = ocr_results.get("back_side_qr", "")

            qr_code_data = (
                front_qr_code_data.strip()
                if front_qr_code_data.strip()
                else back_qr_code_data.strip()
            )

            if isinstance(qr_code_data, tuple):
                qr_code_data = str(qr_code_data)

            # Parse QR code data
            qr_parts = qr_code_data.split("|")
            id_number = qr_parts[0] if len(qr_parts) > 0 else ""
            id_number_old = qr_parts[1] if len(qr_parts) > 1 else ""
            fullname = qr_parts[2] if len(qr_parts) > 2 else ""
            day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
            sex = qr_parts[4] if len(qr_parts) > 4 else ""
            place_of_residence_qr = qr_parts[5] if len(qr_parts) > 5 else ""
            date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""

            # Build the context string with the optimized prompt
            context = (
                "Data extracted from OCR and QR code of the Vietnamese Citizen Identity Card:\n\n"
                "### QR Code Data:\n"
                f"{qr_code_data}\n\n"
                "QR code data format and values:\n"
                f"- ID Number (id_number): {id_number}\n"
                f"- Old ID Number (id_number_old): {id_number_old}\n"
                f"- Full Name (fullname): {fullname}\n"
                f"- Date of Birth (day_of_birth): {day_of_birth} (format: DDMMYYYY)\n"
                f"- Gender (sex): {sex}\n"
                f"- Place of Residence (place_of_residence): {place_of_residence_qr}\n"
                f"- Date of Issue (date_of_issue): {date_of_issue} (format: DDMMYYYY)\n\n"
                "### Front Side OCR Data:\n"
                f"{front_ocr}\n\n"
                "### Back Side OCR Data:\n"
                f"{back_ocr}\n\n"
                "Please analyze and correct the information according to these guidelines:\n\n"
                "1. **Compare and Prioritize Data:**\n"
                "   - Compare information from the QR code and OCR data.\n"
                "   - **Prioritize QR code data** if there are discrepancies.\n\n"
                "2. **Use OCR Data When QR Data is Missing:**\n"
                "   - If information is missing from the QR code or the QR code is empty, **use OCR data**.\n"
                "   - **Pay special attention to fields like `place_of_origin`, `nationality`, and `expiration_date`**, which may not be present in the QR code but are available in the OCR data.\n\n"
                "3. **Correct OCR Errors:**\n"
                "   - Correct spelling mistakes, missing diacritics, missing or incorrect digits in dates, and formatting issues in the OCR data.\n"
                "   - **If a date is incomplete due to OCR errors (e.g., missing digits), attempt to correct it based on context**.\n\n"
                "4. **Date Formatting:**\n"
                "   - Ensure all dates are in **DD/MM/YYYY** format.\n\n"
                "5. **Handling Missing Information:**\n"
                "   - Leave fields empty (`\"\"`) if the information cannot be determined at all.\n\n"
                "6. **Special Cases:**\n"
                "   - **`place_of_origin`**:\n"
                "     - Look for phrases like **\"Quê quán:\", \"Nơi đăng ký khai sinh:\", \"Place of origin:\", \"Place of birth:\"** in both front and back OCR data.\n"
                "     - **Extract and correct the `place_of_origin`** from the OCR data if it's missing in the QR code.\n"
                "   - **`expiration_date`**:\n"
                "     - Identify from phrases like **\"Có giá trị đến:\", \"Date of expiry:\", \"Ngày, tháng, năm hết hạn:\"**.\n"
                "     - **Correct any OCR errors**, such as missing digits in the year.\n"
                "   - **`place_of_residence`**:\n"
                "     - Usually appears after **\"Nơi thường trú:\", \"Place of residence:\", \"Nơi cư trú:\", \"Địa chỉ:\"**.\n"
                "     - May span multiple lines; **combine the lines to form the complete address**.\n\n"
                "7. **Processing Multi-line and Split Information:**\n"
                "   - Handle information that spans multiple lines or is split across different parts.\n"
                "   - **Combine relevant lines** to reconstruct full addresses or other split information.\n\n"
                "8. **Attention to Context:**\n"
                "   - Pay attention to information on unexpected lines.\n"
                "   - **Use contextual clues** to combine or separate information correctly.\n\n"
                "9. **Output Format:**\n"
                "   - **Return ONLY a JSON object** with no additional text, explanations, or comments.\n"
                "   - **Include all fields in the JSON**, even if they are empty (`\"\"`).\n\n"
                "### JSON Structure (values should be in Vietnamese):\n\n"
                "{\n"
                '  "id_number": "",\n'
                '  "id_number_old": "",\n'
                '  "fullname": "",\n'
                '  "day_of_birth": "",\n'
                '  "sex": "",\n'
                '  "nationality": "",\n'
                '  "place_of_residence": "",\n'
                '  "place_of_origin": "",\n'
                '  "expiration_date": "",\n'
                '  "date_of_issue": ""\n'
                "}\n"
            )
            return context

        except Exception as e:
            logger.error(f"Error generating user context: {e}")
            raise Exception("Error generating user context")


    async def send_message(self, image: Optional[Any] = None) -> Dict[str, Any]:
        return await self._send_message(self.system_prompt, self.user_context, image)

    async def send_custom_message(
        self,
        custom_prompt: Optional[str] = None,
        custom_context: Optional[str] = None,
        custom_image: Optional[Any] = None,
    ) -> Dict[str, Any]:
        system_prompt = custom_prompt or self.system_prompt
        user_context = custom_context or self.user_context
        return await self._send_message(system_prompt, user_context, custom_image)

    async def _send_message(
        self, system_prompt: str, user_context: str, image: Optional[Any] = None
    ) -> Dict[str, Any]:
        if not system_prompt or not user_context:
            raise Exception(
                "System prompt và user context phải được thiết lập trước khi gửi tin nhắn."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

        if image:
            image_bytes = convert_image(image)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            messages.append({"role": "user", "content": image_base64})

        return await self.client.chat(self.model, messages=messages)
