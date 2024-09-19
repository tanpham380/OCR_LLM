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

    @staticmethod
    def _get_default_system_prompt() -> str:
        return (
            "Bạn là một trợ lý AI chuyên xử lý văn bản OCR từ thẻ Căn Cước Công Dân Việt Nam (CCCD). "
            "Nhiệm vụ của bạn là phân tích, so sánh, và sửa lỗi OCR từ nhiều nguồn khác nhau như VietOCR, EasyOCR, và PaddleOCR, "
            "đồng thời đối chiếu với dữ liệu từ mã QR để đảm bảo tính chính xác. Vui lòng thực hiện các bước sau:\n"
            "1. So sánh thông tin từ các nguồn OCR với dữ liệu mã QR. Ưu tiên dữ liệu từ mã QR nếu có sự khác biệt.\n"
            "2. Sửa lỗi OCR bao gồm lỗi chính tả, thiếu dấu, hoặc định dạng sai.\n"
            "3. Đảm bảo tất cả ngày tháng được định dạng theo 'DD/MM/YYYY'.\n"
            "4. Để trống ('') nếu thông tin không rõ ràng hoặc thiếu.\n"
            "5. Chỉ chỉnh sửa dữ liệu hiện có; không bổ sung thêm thông tin mới.\n"
            "6. Kết quả trả về phải là JSON, không kèm giải thích hay văn bản thừa."
            "7. Tuyệt đối không trả về bất cứ văn bản nào khác ngoài JSON."
        )

    def set_model(self, model: str) -> None:
        self.model = model

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def set_user_context(self, ocr_results: Dict[str, Any]) -> None:
        """
        Sets the user context by generating the combined context from both front and back OCR and QR data,
        with a fallback to the back QR code if the front QR code is missing.

        Args:
            ocr_results (Dict[str, Any]): Combined OCR results from the front and back images.
        """
        self.user_context = self._generate_user_context(ocr_results)

    def _generate_user_context(self, ocr_results: Dict[str, Any]) -> str:
        """
        Generates the user context from OCR and QR data, including both front and back sides of the ID card.
        Uses back QR data if front QR code is not present.

        Args:
            ocr_results (Dict[str, Any]): Dictionary containing OCR and QR data from the front and back images.

        Returns:
            str: Formatted context string to be passed to the LLM for further processing.
        """
        try:
            front_ocr = ocr_results.get("front_side_ocr", "")
            front_qr_code_data = ocr_results.get("front_side_qr", "")
            back_ocr = ocr_results.get("back_side_ocr", "")
            back_qr_code_data = ocr_results.get("back_side_qr", "")
            if front_qr_code_data or front_qr_code_data == "":
                qr_code_data = front_qr_code_data
            else:
                qr_code_data = back_qr_code_data
            if isinstance(qr_code_data, tuple):
                qr_code_data = str(qr_code_data)
            qr_parts = qr_code_data.split("|") if qr_code_data else [""] * 7
            id_number = qr_parts[0] if len(qr_parts) > 0 else ""
            id_number_old = qr_parts[1] if len(qr_parts) > 1 and qr_parts[1] else " "
            fullname = qr_parts[2] if len(qr_parts) > 2 else ""
            day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
            sex = qr_parts[4] if len(qr_parts) > 4 else ""
            place_of_residence = qr_parts[5] if len(qr_parts) > 5 else ""
            date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""
            context = (
                "Dữ liệu trích xuất từ OCR và mã QR của thẻ CCCD:\n\n"
                "### Dữ liệu từ OCR mặt trước:\n"
                f"{front_ocr}\n\n"
                "### Dữ liệu từ OCR mặt sau:\n"
                f"{back_ocr}\n\n"
                "### Dữ liệu từ mã QR:\n"
                f"{qr_code_data}\n\n"
                "Dữ liệu từ mã QR có định dạng như sau và giá trị lần lượt là:\n"
                f"- Số CCCD (id_number): {id_number}\n"
                f"- CMND (id_number_old): {id_number_old}\n"
                f"- Họ và tên (fullname): {fullname}\n"
                f"- Ngày sinh (day_of_birth): {day_of_birth} (định dạng DDMMYYYY)\n"
                f"- Giới tính (sex): {sex}\n"
                f"- Nơi thường trú (place_of_residence): {place_of_residence}\n"
                f"- Ngày cấp (date_of_issue): {date_of_issue} (định dạng DDMMYYYY)\n\n"
                "Vui lòng phân tích và chỉnh sửa thông tin theo hướng dẫn:\n"
                "1. So sánh dữ liệu từ OCR và mã QR; ưu tiên mã QR nếu khác nhau. Nếu một thông tin không có trong mã QR, hãy sử dụng dữ liệu từ OCR.\n"
                "2. Sửa lỗi chính tả, dấu tiếng Việt, hoặc định dạng sai.\n"
                "3. Đảm bảo ngày tháng theo định dạng DD/MM/YYYY.\n"
                "4. Để trống ('') nếu thông tin hoàn toàn không thể xác định. Nếu thông tin có thể xác định từ dữ liệu OCR hoặc mã QR, hãy sử dụng nó.\n"
                "Xuất kết quả dưới dạng JSON với cấu trúc:\n"
                "{\n"
                '  "id_number": "<Số CCCD | Mã số CCCD | Số định danh cá nhân | Số CMTND | Số CMND | ID number>",\n'
                '  "id_number_old": "<Số CMND | Old ID number>",\n'
                '  "fullname": "<Họ và tên | Full name>",\n'
                '  "day_of_birth": "<Ngày sinh | Date of birth (DD/MM/YYYY)>",\n'
                '  "sex": "<Giới tính | Sex>",\n'
                '  "nationality": "<Quốc tịch | Nationality>",\n'
                '  "place_of_residence": "<Nơi thường trú | Place of residence>",\n'
                '  "place_of_origin": "<Quê quán | Place of origin>",\n'
                '  "expiration_date": "<Ngày hết hạn | Date of expiry (DD/MM/YYYY)>",\n'
                '  "date_of_issue": "<Ngày cấp | Date of issue (DD/MM/YYYY)>"\n'
                "}\n\n"
                "Lưu ý:\n"
                "- Đảm bảo giá trị 'Place of residence' lấy toàn bộ (Ưu tiên từ mã QR), ngay cả khi nó kéo dài qua nhiều dòng hoặc trùng lặp với các thông tin khác.\n"
                "- Khi gặp cụm từ 'Có giá trị đến: [ngày]' hoặc 'Date of expiry', nhận biết đó là 'expiration_date' và tách riêng khỏi 'place_of_residence'.\n"
                "- Bao gồm tất cả các trường trong JSON, ngay cả khi trống.\n"
                "- Không thêm văn bản ngoài JSON.\n"
                "- Ưu tiên độ chính xác, sửa lỗi chính tả và định dạng.\n"
                "- Đảm bảo tất cả ngày tháng theo định dạng DD/MM/YYYY.\n"
                "- Chú ý đến thông tin trên các dòng không mong muốn; kết hợp hoặc tách thông tin đúng theo ngữ cảnh.\n"
                "- Nếu thông tin không có trong mã QR, hãy sử dụng dữ liệu từ OCR.\n"
            )


            # logger.info(f"Generated context: {context}")
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
