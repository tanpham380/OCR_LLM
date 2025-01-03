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




CCCD_FRONT_PROMPT = """
Bạn được cung cấp 1 ảnh mặt trước của thẻ Căn Cước (CC, 12 số, mẫu mới) hoặc Căn Cước Công Dân (CCCD, 12 số, mẫu cũ) hợp pháp.
## Nhiệm vụ
Trích xuất và trả về thông tin OCR của thẻ theo định dạng JSON, bao gồm:
   - Trường cơ bản (cho cả hai loại thẻ): 
       - "số": 12 chữ số hoặc chuỗi số thẻ, Căn Cước là Số Định Danh(Personal Identification Number) , Căn Cước Công Dân là Số(No.)
       - "Họ và tên": Họ và tên đầy đủ dấu tiếng Việt), Căn Cước là Họ, Chữ đệm và tên khai sinh(Full Name), Căn Cước Công Dân là Họ và Tên(Full Name)
       - "Ngày sinh": định dạng DD/MM/YYYY Căn Cước là Ngày, Tháng, Năm sinh(Date of Birth), Căn Cước Công Dân là Ngày sinh(Date of Birth)
       - "Giới tính": chỉ "Nam" hoặc "Nữ" (Sex)
       - "Quốc tịch": thông thường là "Việt Nam"(Nationality)
   - Trường mở rộng (chỉ xuất hiện nếu loại thẻ là "CCCD"):
       - "Quê quán": nếu không có thì trả về "None" (Place of Origin)
       - "Nơi thường trú": nếu không có thì trả về "None" (Place of Residence)
       - "Có giá trị đến": nếu không có thì trả về "None" (Date of Expiry)

## Lưu ý cách phân biệt:
- Thẻ Căn Cước Công Dân (cũ) sẽ hiển thị dòng "Quê quán", "Nơi thường trú", "Có giá trị đến" trên mặt trước. 
- Căn Cước (mẫu mới) không có hoặc không hiển thị dòng "Quê quán", "Nơi thường trú", "Có giá trị đến" trên mặt trước. 

## Tham khảo danh sách các họ phổ biến và tỉnh/thành của Việt Nam:
- Các họ phổ biến ở Việt Nam: NGUYỄN, Nguyễn, TRẦN, Trần, LÊ, Lê, ĐINH, Đinh, PHẠM, Phạm, TRỊNH, Trịnh, LÝ, Lý, HOÀNG, Hoàng, BÙI, Bùi, NGÔ, Ngô, PHAN, Phan, VÕ, Võ, HỒ, Hồ, HUỲNH, Huỳnh, TRƯƠNG, Trương, ĐẶNG, Đặng, ĐỖ, Đỗ, ...
- [Địa danh] Hà Nội, TP. Hồ Chí Minh, Đà Nẵng, Hải Phòng, Cần Thơ, An Giang, Bà Rịa-Vũng Tàu, Bắc Giang, Bắc Kạn, Bạc Liêu, ...
(Tham khảo chi tiết các tỉnh/thành theo danh sách chuẩn của Việt Nam)
- Lưu ý là các thông tin quê quán và dịa chỉ thường trú có thể nằm ở 2 dòng liên tiếp nhau. 
- Không được bỏ sót bất kỳ thông tin chi tiết nào về địa chỉ quê quán hoặc địa chỉ thường trú hoặc ngày hết hạn của thẻ.
- Bảo đảm các câu từ có dấu tiếng Việt là đầy đủ và chính xác.

## Quy tắc kiểm tra và định dạng:
1. "Số": 
   - Đối với CCCD: Bắt buộc là 12 chữ số (ví dụ: 123456789012).
   - Đối với Căn Cước cũ: Có thể là 9 hoặc 12 chữ số, tuân theo định dạng thẻ cũ.
2. "Tên": 
   - Họ và tên phải có đầy đủ dấu tiếng Việt (ví dụ: "Phạm Thanh Tân").
3. "Ngày sinh": 
   - Phải trả về định dạng DD/MM/YYYY (ví dụ: "26/06/2001").
4. "Giới tính": 
   - Chỉ chấp nhận "Nam" hoặc "Nữ".
5. Các trường "Quê quán", "Nơi thường trú", "Có giá trị đến" chỉ bắt buộc cho loại thẻ "CCCD". Nếu không có thông tin thì để "None".
6. "Quốc tịch": 
   - Thông thường là "Việt Nam". Nếu nhận dạng được quốc tịch khác, cần ghi rõ.

Trả lại chính xác kết quả OCR của ảnh qua định dạng JSON như sau:
{
    "Số": "Số thẻ",
    "Tên": "Họ và tên",
    "Ngày sinh": "DD/MM/YYYY Ngày tháng năm sinh",
    "Giới tính": "Nam hoặc Nữ",
    "Quốc tịch": "Việt Nam hoặc quốc tịch khác",
    "Quê quán": " Trích xuất thông tin chi tiết của nơi thường trú. Phải trả lời đầy đủ thông tin nếu có trong ảnh về: địa chỉ nhà, bản, tổ, ấp, thôn, xã, phường, thị trấn, quận, huyện, thị xã, tỉnh, thành phố. Chỉ với CCCD, không có thì None",
    "Nơi thường trú": "Trích xuất thông tin chi tiết của nơi thường trú. Phải trả lời đầy đủ thông tin nếu có trong ảnh về: địa chỉ nhà, bản, tổ, ấp, thôn, xã, phường, thị trấn, quận, huyện, thị xã, tỉnh, thành phố. Chỉ với CCCD, không có thì None",
    "Có giá trị đến": "Ngày hết hạn của giấy tờ này Chỉ với CCCD , không có thì None" 
}

Hãy xuất dữ liệu OCR chính xác và tuân thủ đầy đủ các quy tắc, kể cả khi mặt trước thẻ không hiển thị đầy đủ một số trường (thì trả về None nếu là CCCD). 
"""


CCCD_BACK_PROMPT = """
Bạn được cung cấp 1 ảnh mặt sau của thẻ Căn Cước (CC, 12 số, mẫu mới) hoặc Căn Cước Công Dân (CCCD, 12 số, mẫu cũ) hợp pháp.
## Tham khảo danh sách các họ phổ biến và tỉnh/thành của Việt Nam:
- [Địa danh] Hà Nội, TP. Hồ Chí Minh, Đà Nẵng, Hải Phòng, Cần Thơ, An Giang, Bà Rịa-Vũng Tàu, Bắc Giang, Bắc Kạn, Bạc Liêu, ...
(Tham khảo chi tiết các tỉnh/thành theo danh sách chuẩn của Việt Nam)
## Lưu ý: 
- mặt sau của căn cước công dân không có họ tên hay địa chỉ của người được cấp căn cước, chỉ có thông tin theo thứ tự là các đặc điểm nhân dạng, thể có các dấu vân tay(bên phải) và Ngày, tháng, năm(Date, month, year), nôi cấp (Cục Trưởng Cục Cảnh Sát Quản Lý Hành Chính về Trật Tự Xã Hội - Director General of the police department for administrative management of social order), con dấu , người kí(người cấp) và mã hoá.
- mặt sau của căn cước sẽ có thông tin theo thứ tự là nơi cư trú(Place of residence), nơi đăng kí khai sinh(Place of birth), ngày cấp(DD/MM/YYYY Date of issue), ngày hết hạn(DD/MM/YYYY Date of expiry) ,nơi cấp(Ministry of public security) và mã hoá.
- Phân biệt Căn Cước và Căn Cước Công dân theo mặt sau, không có ảnh chân dung. Thẻ Căn Cước sẽ có mã QRcode và thông tin nơi cấp, ngày cấp..... Còn Căn Cước Công Dân sẽ không có mã QRcode
## Nhiệm vụ
Trả lại chính xác kết quả OCR của ảnh qua định dạng JSON như sau:
{
    "Nơi cư trú": "Trích xuất thông tin chi tiết của nơi cư trú. Phải trả lời đầy đủ thông tin nếu có trong ảnh về: địa chỉ nhà, bản, tổ, ấp, thôn, xã, phường, thị trấn, quận, huyện, thị xã, tỉnh, thành phố. Chỉ với Căn Cước, không có thì None",
    "Nơi đăng kí khai sinh": "Trích xuất thông tin chi tiết của nơi đăng kí khai sinh. Phải trả lời đầy đủ thông tin nếu có trong ảnh về: địa chỉ nhà, bản, tổ, ấp, thôn, xã, phường, thị trấn, quận, huyện, thị xã, tỉnh, thành phố. Chỉ với Căn Cước, không có thì None",
    "Ngày cấp": "Nằm ở mặt sau, không có ảnh chân dung. Ngày, tháng, năm cấp căn cước này",
    "Ngày hết hạn": "Nằm ở mặt sau, không có ảnh chân dung. Ngày hết hạn của thẻ căn cước này, chỉ có với Căn Cước, không có thì None",
    "Nơi cấp": "Nằm ở mặt sau, không có ảnh chân dung. Tên của cơ quan quản lý đóng mộc cấp căn cước này (ví dụ: cục quản lý hành chính về trật tự xã hội,Bộ công an ,...) Trích xuất thông tin chi tiết của nơi cấp. không có thì None",
}
"""