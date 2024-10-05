import base64
from typing import Any, Dict, Optional, Tuple
from app_utils.file_handler import convert_image
from app_utils.logging import get_logger
from config import URL_API_LLM
from ollama import AsyncClient

logger = get_logger(__name__)


class LlmController:
    def __init__(self, model: str = "qwen2.5:14b") -> None:
        self.client = AsyncClient(host=URL_API_LLM)
        self.model = model
        self.system_prompt = self._get_default_system_prompt()
        self.user_context = None
        
    # def _get_default_system_prompt(self) -> str:
    #     return (
    #         "Bạn là một trợ lý AI chuyên xử lý văn bản OCR từ thẻ Căn Cước Công Dân Việt Nam (CCCD). \n"
    #         "Nhiệm vụ của bạn là phân tích, so sánh, và sửa lỗi OCR \n"
    #         "đồng thời đối chiếu với dữ liệu từ mã QR để đảm bảo tính chính xác. Vui lòng thực hiện các bước sau:\n"
    #         "1. So sánh thông tin từ các nguồn OCR với dữ liệu mã QR. Ưu tiên dữ liệu từ mã QR nếu có sự khác biệt.\n"
    #         "2. Sửa lỗi OCR bao gồm lỗi chính tả, thiếu dấu, hoặc định dạng sai.\n"
    #         "3. Đảm bảo tất cả ngày tháng được định dạng theo 'DD/MM/YYYY'.\n"
    #         "4. Để trống ('') nếu thông tin không rõ ràng hoặc thiếu.\n"
    #         "5. Chỉ chỉnh sửa dữ liệu hiện có; không bổ sung thêm thông tin mới.\n"
    #         "6. Kết quả trả về phải là JSON, không kèm giải thích hay văn bản thừa. \n"
    #         "7. Tuyệt đối không trả về bất cứ văn bản nào khác ngoài JSON. \n"
    #     )

    # def _get_default_system_prompt(self) -> str:
    #     return (
    #         "您是一名处理来自越南公民身份证 (CCCD) 的OCR文本的AI助手。\n"
    #         "您的任务是分析、比较和修正OCR错误，同时与二维码数据进行比对以确保准确性。请执行以下步骤：\n"
    #         "1. 比较OCR信息与二维码数据，如果有差异，请优先选择二维码数据。\n"
    #         "2. 修正OCR错误，包括拼写错误、缺少标点或格式错误。\n"
    #         "3. 确保所有日期格式为 'DD/MM/YYYY'。\n"
    #         "4. 如果信息不清楚或缺失，请留空 ('')。\n"
    #         "5. 仅修改现有数据；不要添加新信息。\n"
    #         "6. 返回结果必须是JSON，并且必须使用越南语，不要包含任何解释或多余文本。\n"
    #         "7. 绝对不能返回除JSON以外的任何文本。\n"
    #     )

    def _get_default_system_prompt(self) -> str:
        return (
            "You are an AI assistant specialized in processing OCR text from Vietnamese Citizen ID cards (CCCD). \n"
            "Your task is to analyze, compare, and correct OCR errors, \n"
            "while cross-referencing with data from the QR code to ensure accuracy. Please follow these steps:\n"
            "1. Compare the information from OCR sources with QR code data. Prioritize data from the QR code if there are discrepancies.\n"
            "2. Correct OCR errors, including spelling mistakes, missing diacritics, or formatting errors.\n"
            "3. Ensure all dates are formatted as 'DD/MM/YYYY'.\n"
            "4. Leave fields blank ('') if the information is unclear or missing.\n"
            "5. Only modify existing data; do not add new information.\n"
            "6. The result must be returned in JSON format, without any explanations or extra text. \n"
            "7. Absolutely no text should be returned except for JSON. \n"
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

            # Analyze QR code data
            qr_parts = qr_code_data.split("|")
            id_number = qr_parts[0] if len(qr_parts) > 0 else ""
            id_number_old = qr_parts[1] if len(qr_parts) > 1 else ""
            fullname = qr_parts[2] if len(qr_parts) > 2 else ""
            day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
            sex = qr_parts[4] if len(qr_parts) > 4 else ""
            place_of_residence = qr_parts[5] if len(qr_parts) > 5 else ""
            date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""

            context = (
                "Data extracted from OCR and QR code of the Vietnamese Citizen ID Card (CCCD):\n\n"
                "### Data from front side OCR:\n"
                f"{front_ocr}\n\n"
                "### Data from back side OCR:\n"
                f"{back_ocr}\n\n"
                "### Data from QR code:\n"
                f"{qr_code_data}\n\n"
                "QR code data is formatted as follows, with corresponding values:\n"
                f"- CCCD number (id_number): {id_number}\n"
                f"- Old ID number (id_number_old): {id_number_old}\n"
                f"- Full name (fullname): {fullname}\n"
                f"- Date of birth (day_of_birth): {day_of_birth} (format: DDMMYYYY)\n"
                f"- Gender (sex): {sex}\n"
                f"- Place of residence (place_of_residence): {place_of_residence}\n"
                f"- Date of issue (date_of_issue): {date_of_issue} (format: DDMMYYYY)\n\n"
                "Please analyze and edit the information based on the following instructions:\n"
                "1. Compare data from OCR and QR code; prioritize QR code data in case of discrepancies. If some information is missing in the QR code, use data from OCR.\n"
                "2. Correct spelling, punctuation, or formatting errors in all fields, especially in personal names.\n"
                "Apply Vietnamese capitalization and spelling rules:\n"
                "   - Capitalize the first letter of personal names and place names.\n"
                "   - Correct common mistakes such as confusion between 'l' and 'n', 's' and 'x', 'ch' and 'tr', 'd' and 'gi'.\n"
                "   - Ensure that accent marks are placed in the correct positions.\n"
                "3. Ensure all dates follow the DD/MM/YYYY format.\n"
                "4. Leave blank ('') if information is completely unreadable. If possible, infer information from OCR or QR code.\n"
                "5. Ensure that the value for 'sex' is written out fully (e.g., 'Nam' instead of 'N').\n"
                "6. Fix common OCR errors, such as misidentified characters (e.g., 'Giói tính' to 'Giới tính', 'Ngày sinh/Ide date of birth' to 'Ngày sinh/Date of birth').\n"
                "7. For phrases like 'Có giá trị đến: [date]' or 'Date of expiry', recognize that this refers to 'date_of_expiration' and separate it from 'place_of_residence'. 'Date of expiry' may appear on either the front or back of the CCCD.\n"
                "8. If there is one date on the back of the CCCD, it is the 'date_of_issue'. If there are two dates, they are 'date_of_issue' and 'date_of_expiration'; the issue date will precede the expiration date.\n"
                "10. When extracting 'place_of_origin' and 'place_of_residence', ensure they are ordered from the smallest to largest administrative unit.\n"
                "11. Ensure 'nationality' is spelled correctly and with full Vietnamese accents for any country (e.g., 'Việt Nam' instead of 'Vietnam').\n"
                "12. For 'place_of_issue', extract the value from the OCR data as follows:\n"
                "    - If 'Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội' appears in the OCR data, use it as 'place_of_issue'.\n"
                "    - If not, use 'Bộ Công An'.\n"
                "13. Combine information from all sources to fill out the following 11 fields completely.\n"
                "14. Do not include any explanations, just return the result in JSON format.\n"
                "15. Prioritize accuracy, spelling corrections, and formatting.\n"
                "16. Ensure all dates are in DD/MM/YYYY format.\n"
                "17. Be mindful of unwanted lines; combine or separate information based on context.\n"
                "18. If information is not available in the QR code, use data from OCR.\n"
                "19. Correct misidentifications of personal names, places, and technical terms.\n"
                "20. Ensure the value for 'sex' is either 'Nam' or 'Nữ'.\n"
                "21. Although this prompt is in English, please provide the final JSON response in Vietnamese.\n\n"
                "Return the result in Vietnamese and ensure it contains complete information for both 'nơi thường trú' (place_of_residence) and 'quê quán | Nơi Sinh' (place_of_origin)"
                "Return the result as JSON with the following 11 fields:\n"
                "{\n"
                '  "id_number": "<Số CCCD>",\n'
                '  "id_number_old": "<Số CMND cũ>",\n'
                '  "fullname": "<Họ và tên>",\n'
                '  "day_of_birth": "<Ngày sinh (DD/MM/YYYY)>",\n'
                '  "sex": "<Giới tính>",\n'
                '  "nationality": "<Quốc tịch>",\n'
                '  "place_of_residence": "<Nơi thường trú>",\n'
                '  "place_of_origin": "<Quê quán , Nơi Sinh >",\n'
                '  "date_of_expiration": "<Ngày hết hạn (DD/MM/YYYY)>",\n'
                '  "date_of_issue": "<Ngày cấp (DD/MM/YYYY)>",\n'
                '  "place_of_issue": "<Nơi cấp>"\n'
                "}\n"
            )
            print(context)
            return context

        except Exception as e:
            logger.error(f"Error generating user context: {e}")
            raise Exception("Error generating user context")
