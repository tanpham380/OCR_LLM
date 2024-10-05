from typing import Any, Dict


def generate_user_context( ocr_results: Dict[str, Any]) -> str:
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

        # Phân tích dữ liệu QR code
        qr_parts = qr_code_data.split("|")
        id_number = qr_parts[0] if len(qr_parts) > 0 else ""
        id_number_old = qr_parts[1] if len(qr_parts) > 1 else ""
        fullname = qr_parts[2] if len(qr_parts) > 2 else ""
        day_of_birth = qr_parts[3] if len(qr_parts) > 3 else ""
        sex = qr_parts[4] if len(qr_parts) > 4 else ""
        place_of_residence = qr_parts[5] if len(qr_parts) > 5 else ""
        date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""
        
        context = (
            "Dữ liệu trích xuất từ OCR và QR code của Căn cước công dân (CCCD):\n\n"
            "### Dữ liệu từ OCR mặt trước:\n"
            f"{front_ocr}\n"
            "### Dữ liệu từ OCR mặt sau:\n"
            f"{back_ocr}\n"
            "### Dữ liệu từ QR code:\n"
            f"{qr_code_data}\n"
            "Dữ liệu QR code được định dạng như sau, với các giá trị tương ứng:\n"
            f"- Số CCCD (id_number): {id_number}\n"
            f"- Số CMND cũ (id_number_old): {id_number_old}\n"
            f"- Họ và tên (fullname): {fullname}\n"
            f"- Ngày sinh (day_of_birth): {day_of_birth} (định dạng: DDMMYYYY)\n"
            f"- Giới tính (sex): {sex}\n"
            f"- Nơi thường trú (place_of_residence): {place_of_residence}\n"
            f"- Ngày cấp (date_of_issue): {date_of_issue} (định dạng: DDMMYYYY)\n\n"
            "Vui lòng phân tích và chỉnh sửa thông tin dựa trên các hướng dẫn sau:\n"
            "1. So sánh dữ liệu từ OCR và QR code; ưu tiên dữ liệu QR code trong trường hợp có sự khác biệt. Nếu thiếu thông tin trong QR code, sử dụng dữ liệu từ OCR.\n"
            "2. Đảm bảo tất cả các ngày đều theo định dạng DD/MM/YYYY.\n"
            "3. Để trống ('') nếu thông tin không thể đọc được hoàn toàn. Nếu có thể, suy luận thông tin từ OCR hoặc QR code.\n"
            "4. Đảm bảo giá trị 'giới tính' được viết đầy đủ (ví dụ: 'Nam' thay vì 'N').\n"
            "5. Sửa các lỗi OCR thông thường, chẳng hạn như ký tự bị nhận diện sai (ví dụ: 'Giói tính' thành 'Giới tính', 'Ngày sinh/Ide date of birth' thành 'Ngày sinh/Date of birth').\n"
            "6. Đối với các cụm từ như 'Có giá trị đến: [ngày]' hoặc 'Date of expiry', hiểu rằng đây là 'ngày hết hạn' và tách nó khỏi 'nơi thường trú'. 'Ngày hết hạn' có thể xuất hiện trên cả mặt trước hoặc mặt sau của CCCD.\n"
            "7. Nếu có một ngày trên mặt sau của CCCD, đó là 'ngày cấp'. Nếu có hai ngày, đó là 'ngày cấp' và 'ngày hết hạn'; ngày cấp sẽ trước ngày hết hạn.\n"
            "8. Hiểu và sắp xếp các đơn vị hành chính Việt Nam từ nhỏ đến lớn: Thôn/Làng/Khu phố < Xã/Phường/Thị trấn < Quận/Huyện/Thị xã/Thành phố thuộc tỉnh < Tỉnh/Thành phố trực thuộc trung ương.\n"
            "9 Khi trích xuất 'quê quán' và 'nơi thường trú', đảm bảo chúng được sắp xếp từ đơn vị hành chính nhỏ nhất đến lớn nhất.\n"
            "12. Đảm bảo 'quốc tịch' được viết đúng chính tả và đầy đủ dấu tiếng Việt cho bất kỳ quốc gia nào (ví dụ: 'Việt Nam' thay vì 'Vietnam').\n"
            "13. Đối với 'nơi cấp', trích xuất giá trị từ dữ liệu OCR như sau:\n"
            "    - Nếu có cụm từ 'Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội' xuất hiện trong dữ liệu OCR, sử dụng nó làm 'nơi cấp'.\n"
            "    - Nếu không có, sử dụng 'Bộ Công An'.\n"
            "14. Kết hợp thông tin từ tất cả các nguồn để điền đủ 11 trường.\n"
            "17. Đảm bảo tất cả các ngày đều theo định dạng DD/MM/YYYY.\n"
            "19. Nếu không có thông tin trong QR code, sử dụng dữ liệu từ OCR.\n"
            "Trả về kết quả  một định dạng JSON với 11 trường sau:\n"
            "{\n"
            '  "id_number": "<Số CCCD>",\n'
            '  "id_number_old": "<Số CMND cũ>",\n'
            '  "fullname": "<Họ và tên>",\n'
            '  "day_of_birth": "<Ngày sinh (DD/MM/YYYY)>",\n'
            '  "sex": "<Giới tính>",\n'
            '  "nationality": "<Quốc tịch>",\n'
            '  "place_of_residence": "<Nơi thường trú>",\n'
            '  "place_of_origin": "<Quê quán>",\n'
            '  "date_of_expiration": "<Ngày hết hạn (DD/MM/YYYY)>",\n'
            '  "date_of_issue": "<Ngày cấp (DD/MM/YYYY)>",\n'
            '  "place_of_issue": "<Nơi cấp>"\n'
            "}\n"
        )
        
        print(context)
        return context

    except Exception as e:
        raise Exception("Lỗi tạo ngữ cảnh người dùng")
