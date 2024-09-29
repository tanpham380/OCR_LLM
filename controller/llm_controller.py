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


    # def _generate_user_context(self, ocr_results: Dict[str, Any]) -> str:
    #     try:
    #         front_ocr = (
    #             ocr_results.get("front_side_ocr", {})
    #             .get("package_ocr", "")
    #             .replace("\\", "")
    #         )
    #         back_ocr = (
    #             ocr_results.get("back_side_ocr", {})
    #             .get("package_ocr", "")
    #             .replace("\\", "")
    #         )

    #         front_qr_code_data = ocr_results.get("front_side_qr", "")
    #         back_qr_code_data = ocr_results.get("back_side_qr", "")

    #         qr_code_data = (
    #             front_qr_code_data.strip()
    #             if front_qr_code_data.strip()
    #             else back_qr_code_data.strip()
    #         )
    #         if isinstance(qr_code_data, tuple):
    #             qr_code_data = str(qr_code_data)

    #         # Phân tích dữ liệu mã QR
    #         qr_parts = qr_code_data.split("|")
    #         id_number = qr_parts[0] if len(qr_parts) > 0 else ""
    #         id_number_old = qr_parts[1] if len(qr_parts) > 1 else ""
    #         fullname = qr_parts[2] if len(qr_parts) > 2 else ""
    #         day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
    #         sex = qr_parts[4] if len(qr_parts) > 4 else ""
    #         place_of_residence = qr_parts[5] if len(qr_parts) > 5 else ""
    #         date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""

    #         context = (
    #             "Dữ liệu trích xuất từ OCR và mã QR của thẻ CCCD:\n\n"
    #             "### Dữ liệu từ OCR mặt trước:\n"
    #             f"{front_ocr}\n\n"
    #             "### Dữ liệu từ OCR mặt sau:\n"
    #             f"{back_ocr}\n\n"
    #             "### Dữ liệu từ mã QR:\n"
    #             f"{qr_code_data}\n\n"
    #             "Dữ liệu từ mã QR có định dạng như sau và giá trị lần lượt là:\n"
    #             f"- Số CCCD (id_number): {id_number}\n"
    #             f"- CMND (id_number_old): {id_number_old}\n"
    #             f"- Họ và tên (fullname): {fullname}\n"
    #             f"- Ngày sinh (day_of_birth): {day_of_birth} (định dạng DDMMYYYY)\n"
    #             f"- Giới tính (sex): {sex}\n"
    #             f"- Nơi thường trú (place_of_residence): {place_of_residence}\n"
    #             f"- Ngày cấp (date_of_issue): {date_of_issue} (định dạng DDMMYYYY)\n\n"
    #             "Vui lòng phân tích và chỉnh sửa thông tin theo hướng dẫn:\n"
    #             "1. So sánh dữ liệu từ OCR và mã QR; ưu tiên mã QR nếu khác nhau. Nếu một thông tin không có trong mã QR, hãy sử dụng dữ liệu từ OCR.\n"
    #             "2. Sửa lỗi chính tả, dấu tiếng Việt, hoặc định dạng sai.\n"
    #             "3. Đảm bảo ngày tháng theo định dạng DD/MM/YYYY.\n"
    #             "4. Để trống ('') nếu thông tin hoàn toàn không thể xác định. Nếu thông tin có thể xác định từ dữ liệu OCR hoặc mã QR, hãy sử dụng nó.\n"
    #             "Xuất kết quả dưới dạng JSON với cấu trúc 10 trường dưới đây:\n"
    #             "{\n"
    #             '  "id_number": "<Số CCCD | Mã số CCCD | Số định danh cá nhân | Số CMTND | Số CMND | ID number>",\n'
    #             '  "id_number_old": "<Số CMND | Old ID number>",\n'
    #             '  "fullname": "<Họ và tên | Full name>",\n'
    #             '  "day_of_birth": "<Ngày sinh | Date of birth (DD/MM/YYYY)>",\n'
    #             '  "sex": "<Giới tính | Sex>",\n'
    #             '  "nationality": "<Quốc tịch | Nationality>",\n'
    #             '  "place_of_residence": "<Nơi thường trú | Place of residence>",\n'
    #             '  "place_of_origin": "<Quê quán | Place of origin>",\n'
    #             '  "date_of_expiration": "<Ngày hết hạn | Date of expiry (DD/MM/YYYY)>",\n'
    #             '  "date_of_issue": "<Ngày cấp | Date of issue (DD/MM/YYYY)>"\n'
    #             "}\n\n"
    #             "Lưu ý:\n"
    #             "- Đảm bảo giá trị 'Place of residence' lấy toàn bộ (ưu tiên từ mã QR), ngay cả khi nó kéo dài qua nhiều dòng hoặc trùng lặp với các thông tin khác.\n"
    #             "- Khi gặp cụm từ 'Có giá trị đến: [ngày]' hoặc 'Date of expiry', nhận biết đó là 'date_of_expiration' và tách riêng khỏi 'place_of_residence'. 'Date of expiry' có thể xuất hiện ở mặt trước hoặc mặt sau của CCCD.\n"
    #             "- Nếu mặt sau của CCCD có 1 ngày tháng năm, đó là 'date_of_issue'. Nếu có 2 ngày, thì đó là 'date_of_issue' và 'date_of_expiration'; ngày cấp sẽ nhỏ hơn ngày hết hạn.\n"
    #             "- Bao gồm tất cả các trường trong JSON, ngay cả khi trống.\n"
    #             "- Không giải thích gì thêm, văn bản ngoài JSON.\n"
    #             "- Ưu tiên độ chính xác, sửa lỗi chính tả và định dạng.\n"
    #             "- Đảm bảo tất cả ngày tháng theo định dạng DD/MM/YYYY.\n"
    #             "- Chú ý đến thông tin trên các dòng không mong muốn; kết hợp hoặc tách thông tin đúng theo ngữ cảnh.\n"
    #             "- Nếu thông tin không có trong mã QR, hãy sử dụng dữ liệu từ OCR.\n"
    #             "- Hiểu và sắp xếp các cấp bậc hành chính ở Việt Nam theo thứ tự từ nhỏ đến lớn: Thôn/Ấp/Khu phố < Xã/Phường/Thị trấn < Quận/Huyện/Thị xã/Thành phố thuộc tỉnh < Tỉnh/Thành phố trực thuộc trung ương.\n"
    #             "- Khi xuất 'place_of_origin' và 'place_of_residence', hãy đảm bảo sắp xếp theo thứ tự cấp bậc hành chính từ nhỏ đến lớn.\n"
    #             "- Lưu ý rằng 'nationality' có thể lấy từ dữ liệu OCR nếu không có trong mã QR.\n"
    #             "- Chú ý các dòng MRZ (Machine Readable Zone) ở cuối mặt sau (các dòng có dạng 'IDVNM...'); nếu cần thiết, sử dụng để xác định 'id_number' và các thông tin khác.\n"
    #         )
    #         return context

    #     except Exception as e:
    #         logger.error(f"Lỗi tạo context người dùng: {e}")
    #         raise Exception("Lỗi tạo context người dùng")

    # def _generate_user_context(self, ocr_results: Dict[str, Any]) -> str:
    #     try:
    #         front_ocr = (
    #             ocr_results.get("front_side_ocr", {})
    #             .get("package_ocr", "")
    #             .replace("\\", "")
    #         )
    #         back_ocr = (
    #             ocr_results.get("back_side_ocr", {})
    #             .get("package_ocr", "")
    #             .replace("\\", "")
    #         )

    #         front_qr_code_data = ocr_results.get("front_side_qr", "")
    #         back_qr_code_data = ocr_results.get("back_side_qr", "")

    #         qr_code_data = (
    #             front_qr_code_data.strip()
    #             if front_qr_code_data.strip()
    #             else back_qr_code_data.strip()
    #         )
    #         if isinstance(qr_code_data, tuple):
    #             qr_code_data = str(qr_code_data)

    #         # Analyze QR code data
    #         qr_parts = qr_code_data.split("|")
    #         id_number = qr_parts[0] if len(qr_parts) > 0 else ""
    #         id_number_old = qr_parts[1] if len(qr_parts) > 1 else ""
    #         fullname = qr_parts[2] if len(qr_parts) > 2 else ""
    #         day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
    #         sex = qr_parts[4] if len(qr_parts) > 4 else ""
    #         place_of_residence = qr_parts[5] if len(qr_parts) > 5 else ""
    #         date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""

    #         context = (
    #             "Data extracted from OCR and the QR code of the Vietnamese Citizen ID card (CCCD):\n\n"
    #             "### Data from front-side OCR:\n"
    #             f"{front_ocr}\n\n"
    #             "### Data from back-side OCR:\n"
    #             f"{back_ocr}\n\n"
    #             "### Data from the QR code:\n"
    #             f"{qr_code_data}\n\n"
    #             "The data from the QR code is formatted as follows, with the values corresponding to:\n"
    #             f"- ID number (id_number): {id_number}\n"
    #             f"- Old ID number (id_number_old): {id_number_old}\n"
    #             f"- Full name (fullname): {fullname}\n"
    #             f"- Date of birth (day_of_birth): {day_of_birth} (format DDMMYYYY)\n"
    #             f"- Sex (sex): {sex}\n"
    #             f"- Place of residence (place_of_residence): {place_of_residence}\n"
    #             f"- Date of issue (date_of_issue): {date_of_issue} (format DDMMYYYY)\n\n"
    #             "Please analyze and correct the information based on the following guidelines:\n"
    #             "1. Compare the data from OCR and the QR code; prioritize the QR code if there's a discrepancy. If some information is missing from the QR code, use the data from OCR.\n"
    #             "2. Correct spelling, Vietnamese diacritical marks, or formatting errors in all fields.\n"
    #             "3. Ensure all dates follow the DD/MM/YYYY format.\n"
    #             "4. Leave blank ('') if the information is completely indiscernible. If any information can be inferred from the OCR or QR code data, use it.\n"
    #             "Return the result as JSON, structured with the following 10 fields:\n"
    #             "{\n"
    #             '  "id_number": "<Số CCCD | Mã số CCCD | Số định danh cá nhân | Số CMTND | Số CMND | ID number>",\n'
    #             '  "id_number_old": "<Số CMND | Old ID number>",\n'
    #             '  "fullname": "<Họ và tên | Full name>",\n'
    #             '  "day_of_birth": "<Ngày sinh | Date of birth (DD/MM/YYYY)>",\n'
    #             '  "sex": "<Giới tính | Sex>",\n'
    #             '  "nationality": "<Quốc tịch | Nationality>",\n'
    #             '  "place_of_residence": "<Nơi thường trú | Place of residence>",\n'
    #             '  "place_of_origin": "<Quê quán | Place of origin>",\n'
    #             '  "date_of_expiration": "<Ngày hết hạn | Date of expiry (DD/MM/YYYY)>",\n'
    #             '  "date_of_issue": "<Ngày cấp | Date of issue (DD/MM/YYYY)>"\n'
    #             "}\n\n"
    #             "Notes:\n"
    #             "- Ensure that the value for 'Place of residence' is complete (prioritize the QR code), even if it spans multiple lines or duplicates other information.\n"
    #             "- When encountering phrases like 'Có giá trị đến: [ngày]' or 'Date of expiry', recognize that this is the 'date_of_expiration' and separate it from 'place_of_residence'. 'Date of expiry' can appear on the front or back of the CCCD.\n"
    #             "- If the back of the CCCD has one date, it's the 'date_of_issue'. If there are two dates, they are the 'date_of_issue' and 'date_of_expiration'; the issue date will be earlier than the expiry date.\n"
    #             "- Include all fields in the JSON, even if they are empty.\n"
    #             "- Do not include any explanations, just JSON text.\n"
    #             "- Prioritize accuracy, correct spelling, and formatting.\n"
    #             "- Ensure all dates are in the DD/MM/YYYY format.\n"
    #             "- Pay attention to unwanted lines; combine or split information according to the context.\n"
    #             "- If the information is not in the QR code, use the OCR data.\n"
    #             "- Understand and arrange the administrative divisions in Vietnam from smallest to largest: Hamlet/Neighborhood < Commune/Ward/Town < District/Town/Provincial City < Province/Central City.\n"
    #             "- When outputting 'place_of_origin' and 'place_of_residence', ensure they are arranged in order from the smallest to largest administrative levels.\n"
    #             "- Ensure that 'nationality' is correctly spelled with proper diacritical marks for any country (e.g., 'Việt Nam' instead of 'Vietnam', 'Cộng hòa Séc' instead of 'Czech Republic').\n"
    #             "- Pay attention to the MRZ (Machine Readable Zone) lines on the back (lines like 'IDVNM...'); use them to identify 'id_number' and other information if necessary.\n"
    #         )
    #         return context

    #     except Exception as e:
    #         logger.error(f"Error generating user context: {e}")
    #         raise Exception("Error generating user context")
    def _generate_user_context(self, ocr_results: Dict[str, Any]) -> str:
        try:
            import re

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

            # Extract MRZ data from back_ocr
            # MRZ typically consists of two or three lines of uppercase letters, numbers, and '<' symbols
            mrz_pattern = re.compile(r'^[A-Z0-9<]{30,}$', re.MULTILINE)
            mrz_matches = mrz_pattern.findall(back_ocr)
            mrz_data = "\n".join(mrz_matches)

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
                "Data extracted from OCR and the QR code of the Vietnamese Citizen ID card (CCCD):\n\n"
                "### Data from front-side OCR:\n"
                f"{front_ocr}\n\n"
                "### Data from back-side OCR:\n"
                f"{back_ocr}\n\n"
                "### MRZ data extracted from back-side OCR:\n"
                f"{mrz_data}\n\n"
                "### Data from the QR code:\n"
                f"{qr_code_data}\n\n"
                "The data from the QR code is formatted as follows, with the values corresponding to:\n"
                f"- ID number (id_number): {id_number}\n"
                f"- Old ID number (id_number_old): {id_number_old}\n"
                f"- Full name (fullname): {fullname}\n"
                f"- Date of birth (day_of_birth): {day_of_birth} (format DDMMYYYY)\n"
                f"- Sex (sex): {sex}\n"
                f"- Place of residence (place_of_residence): {place_of_residence}\n"
                f"- Date of issue (date_of_issue): {date_of_issue} (format DDMMYYYY)\n\n"
                "Please analyze and correct the information based on the following guidelines:\n"
                "1. Compare the data from OCR, MRZ, and the QR code; prioritize the QR code if there's a discrepancy. If some information is missing from the QR code, use the data from MRZ or OCR in that order.\n"
                "2. Correct spelling, Vietnamese diacritical marks, or formatting errors in all fields.\n"
                "3. Ensure all dates follow the DD/MM/YYYY format.\n"
                "4. Leave blank ('') if the information is completely indiscernible. If any information can be inferred from the OCR, MRZ, or QR code data, use it.\n"
                "Return the result as JSON, structured with the following 10 fields:\n"
                "{\n"
                '  "id_number": "<Số CCCD | Mã số CCCD | Số định danh cá nhân | Số CMTND | Số CMND | ID number>",\n'
                '  "id_number_old": "<Số CMND | Old ID number>",\n'
                '  "fullname": "<Họ và tên | Full name>",\n'
                '  "day_of_birth": "<Ngày sinh | Date of birth (DD/MM/YYYY)>",\n'
                '  "sex": "<Giới tính | Sex>",\n'
                '  "nationality": "<Quốc tịch | Nationality>",\n'
                '  "place_of_residence": "<Nơi thường trú | Place of residence>",\n'
                '  "place_of_origin": "<Quê quán | Place of origin>",\n'
                '  "date_of_expiration": "<Ngày hết hạn | Date of expiry (DD/MM/YYYY)>",\n'
                '  "date_of_issue": "<Ngày cấp | Date of issue (DD/MM/YYYY)>"\n'
                "}\n\n"
                "Notes:\n"
                "- Ensure that the value for 'Place of residence' is complete (prioritize the QR code), even if it spans multiple lines or duplicates other information.\n"
                "- When encountering phrases like 'Có giá trị đến: [ngày]' or 'Date of expiry', recognize that this is the 'date_of_expiration' and separate it from 'place_of_residence'. 'Date of expiry' can appear on the front or back of the CCCD.\n"
                "- If the back of the CCCD has one date, it's the 'date_of_issue'. If there are two dates, they are the 'date_of_issue' and 'date_of_expiration'; the issue date will be earlier than the expiry date.\n"
                "- Include all fields in the JSON, even if they are empty.\n"
                "- Do not include any explanations, just JSON text.\n"
                "- Prioritize accuracy, correct spelling, and formatting.\n"
                "- Ensure all dates are in the DD/MM/YYYY format.\n"
                "- Pay attention to unwanted lines; combine or split information according to the context.\n"
                "- If the information is not in the QR code, use the MRZ data or OCR data.\n"
                "- Understand and arrange the administrative divisions in Vietnam from smallest to largest: Hamlet/Neighborhood < Commune/Ward/Town < District/Town/Provincial City < Province/Central City.\n"
                "- When outputting 'place_of_origin' and 'place_of_residence', ensure they are arranged in order from the smallest to largest administrative levels.\n"
                "- Ensure that 'nationality' is correctly spelled with proper diacritical marks for any country (e.g., 'Việt Nam' instead of 'Vietnam', 'Cộng hòa Séc' instead of 'Czech Republic').\n"
                "- Use the MRZ data to extract 'id_number', 'fullname', 'day_of_birth', 'sex', and 'nationality' if they are missing from the QR code and OCR data.\n"
            )
            return context

        except Exception as e:
            logger.error(f"Error generating user context: {e}")
            raise Exception("Error generating user context")
