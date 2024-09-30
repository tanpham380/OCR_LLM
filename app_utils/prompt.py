from typing import Any, Dict


def generate_user_context(ocr_results: Dict[str, Any]) -> str:
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

        # Phân tích dữ liệu mã QR
        qr_parts = qr_code_data.split("|")
        id_number = qr_parts[0] if len(qr_parts) > 0 else ""
        id_number_old = qr_parts[1] if len(qr_parts) > 1 else ""
        fullname = qr_parts[2] if len(qr_parts) > 2 else ""
        day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
        sex = qr_parts[4] if len(qr_parts) > 4 else ""
        place_of_residence = qr_parts[5] if len(qr_parts) > 5 else ""
        date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""

        context = (
            "Dữ liệu được trích xuất từ OCR và mã QR của thẻ Căn cước công dân (CCCD):\n\n"
            "### Dữ liệu từ OCR mặt trước:\n"
            f"{front_ocr}\n\n"
            "### Dữ liệu từ OCR mặt sau:\n"
            f"{back_ocr}\n\n"
            "### Dữ liệu từ mã QR:\n"
            f"{qr_code_data}\n\n"
            "Dữ liệu từ mã QR được định dạng như sau, với các giá trị tương ứng:\n"
            f"- Số CCCD (id_number): {id_number}\n"
            f"- Số CMND cũ (id_number_old): {id_number_old}\n"
            f"- Họ và tên (fullname): {fullname}\n"
            f"- Ngày sinh (day_of_birth): {day_of_birth} (định dạng DDMMYYYY)\n"
            f"- Giới tính (sex): {sex}\n"
            f"- Nơi thường trú (place_of_residence): {place_of_residence}\n"
            f"- Ngày cấp (date_of_issue): {date_of_issue} (định dạng DDMMYYYY)\n\n"
            "Vui lòng phân tích và chỉnh sửa thông tin dựa trên các hướng dẫn sau:\n"
            "1. So sánh dữ liệu từ OCR và mã QR; ưu tiên mã QR nếu có sự khác biệt. Nếu một số thông tin thiếu từ mã QR, hãy sử dụng dữ liệu từ OCR.\n"
            "2. Chỉnh sửa chính tả, dấu câu tiếng Việt, hoặc các lỗi định dạng ở tất cả các trường.\n"
            "3. Đảm bảo tất cả các ngày theo định dạng DD/MM/YYYY.\n"
            "4. Để trống ('') nếu thông tin hoàn toàn không rõ ràng. Nếu có thể suy luận thông tin từ OCR hoặc mã QR, hãy sử dụng nó.\n"
            "5. Đảm bảo giá trị của 'giới tính' được viết đầy đủ (ví dụ: 'Nam' thay vì 'N').\n"
            "Trả về kết quả dưới dạng JSON, được cấu trúc với 10 trường sau:\n"
            "{\n"
            '  "id_number": "<Số CCCD | Mã số CCCD | Số định danh cá nhân | Số CMTND | Số CMND>",\n'
            '  "id_number_old": "<Số CMND cũ>",\n'
            '  "fullname": "<Họ và tên>",\n'
            '  "day_of_birth": "<Ngày sinh (DD/MM/YYYY)>",\n'
            '  "sex": "<Giới tính>",\n'
            '  "nationality": "<Quốc tịch>",\n'
            '  "place_of_residence": "<Nơi thường trú>",\n'
            '  "place_of_origin": "<Quê quán | Nơi Đăng ký Khai sinh>",\n'
            '  "date_of_expiration": "<Ngày hết hạn (DD/MM/YYYY)>",\n'
            '  "date_of_issue": "<Ngày cấp (DD/MM/YYYY)>",\n'
            '  "place_of_issue": "<Nơi cấp>"\n'
            "}\n\n"
            "Ghi chú:\n"
            "- Đảm bảo rằng giá trị 'Nơi thường trú' đầy đủ (ưu tiên mã QR), ngay cả khi nó nằm trên nhiều dòng hoặc trùng lặp thông tin khác.\n"
            "- Khi gặp các cụm từ như 'Có giá trị đến: [ngày]' hoặc 'Date of expiry', hãy nhận ra rằng đây là 'ngày hết hạn' và tách nó khỏi 'nơi thường trú'. 'Ngày hết hạn' có thể xuất hiện ở mặt trước hoặc mặt sau của CCCD.\n"
            "- Nếu mặt sau của CCCD có một ngày, đó là 'ngày cấp'. Nếu có hai ngày, chúng là 'ngày cấp' và 'ngày hết hạn'; ngày cấp sẽ sớm hơn ngày hết hạn.\n"
            "- Bao gồm tất cả các trường trong JSON, ngay cả khi chúng trống.\n"
            "- Không bao gồm bất kỳ giải thích nào, chỉ văn bản JSON.\n"
            "- Ưu tiên độ chính xác, chính tả đúng và định dạng.\n"
            "- Đảm bảo tất cả các ngày theo định dạng DD/MM/YYYY.\n"
            "- Chú ý đến các dòng không mong muốn; kết hợp hoặc tách thông tin theo ngữ cảnh.\n"
            "- Nếu thông tin không có trong mã QR, sử dụng dữ liệu OCR.\n"
            "- Hiểu và sắp xếp các đơn vị hành chính tại Việt Nam từ nhỏ đến lớn: Thôn/Khối phố < Xã/Phường/Thị trấn < Huyện/Thị xã/Thành phố trực thuộc tỉnh < Tỉnh/Thành phố trực thuộc Trung ương.\n"
            "- Khi xuất ra 'nơi sinh' và 'nơi thường trú', hãy đảm bảo sắp xếp theo thứ tự từ nhỏ đến lớn theo các cấp hành chính.\n"
            "- Đảm bảo rằng 'quốc tịch' được viết đúng chính tả với các dấu câu đúng cho bất kỳ quốc gia nào (ví dụ: 'Việt Nam' thay vì 'Vietnam', 'Cộng hòa Séc' thay vì 'Czech Republic').\n"
            "- Nơi cấp bắt đầu với 'Bộ Công An' hoặc 'CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI'.\n"
        )
        return context

    except Exception as e:
        raise Exception("Lỗi khi tạo ngữ cảnh người dùng")
