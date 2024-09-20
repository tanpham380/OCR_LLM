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
#         place_of_residence_qr = qr_parts[5] if len(qr_parts) > 5 else ""
#         date_of_issue = qr_parts[6] if len(qr_parts) > 6 else ""

#         # Xây dựng chuỗi context với prompt tối ưu
#         context = (
#             "Dữ liệu trích xuất từ OCR và mã QR của Căn cước công dân Việt Nam:\n\n"
#             "### Dữ liệu Mã QR:\n"
#             f"{qr_code_data}\n\n"
#             "Định dạng và giá trị của dữ liệu mã QR:\n"
#             f"- Số Căn cước công dân (id_number): {id_number}\n"
#             f"- Số CMND cũ (id_number_old): {id_number_old}\n"
#             f"- Họ và tên (fullname): {fullname}\n"
#             f"- Ngày sinh (day_of_birth): {day_of_birth} (định dạng: DDMMYYYY)\n"
#             f"- Giới tính (sex): {sex}\n"
#             f"- Nơi thường trú (place_of_residence): {place_of_residence_qr}\n"
#             f"- Ngày cấp (date_of_issue): {date_of_issue} (định dạng: DDMMYYYY)\n\n"
#             "### Dữ liệu OCR Mặt Trước:\n"
#             f"{front_ocr}\n\n"
#             "### Dữ liệu OCR Mặt Sau:\n"
#             f"{back_ocr}\n\n"
#             "Hãy phân tích và chỉnh sửa thông tin theo các hướng dẫn sau:\n\n"
#             "1. **So sánh và Ưu tiên Dữ liệu:**\n"
#             "   - So sánh thông tin từ mã QR và dữ liệu OCR.\n"
#             "   - **Ưu tiên dữ liệu mã QR** nếu có sự không đồng nhất.\n\n"
#             "2. **Sử dụng Dữ liệu OCR Khi Mã QR Thiếu:**\n"
#             "   - Nếu thông tin thiếu từ mã QR hoặc mã QR trống, **sử dụng dữ liệu OCR**.\n"
#             "   - **Chú ý đặc biệt tới các trường như `place_of_origin`, `nationality`, và `expiration_date`**, có thể không có trong mã QR nhưng có trong dữ liệu OCR.\n\n"
#             "3. **Chỉnh sửa Lỗi OCR:**\n"
#             "   - Chỉnh sửa lỗi chính tả, thiếu dấu, thiếu hoặc sai số trong ngày tháng và các vấn đề định dạng trong dữ liệu OCR.\n"
#             "   - **Nếu một ngày bị thiếu do lỗi OCR (ví dụ, thiếu số), hãy cố gắng chỉnh sửa dựa trên ngữ cảnh**.\n\n"
#             "4. **Định dạng Ngày:**\n"
#             "   - Đảm bảo tất cả các ngày đều có định dạng **DD/MM/YYYY**.\n\n"
#             "5. **Xử lý Thông tin Thiếu:**\n"
#             "   - Để trống các trường (`\"\"`) nếu không thể xác định thông tin.\n\n"
#             "6. **Trường Hợp Đặc Biệt:**\n"
#             "   - **`place_of_origin`**:\n"
#             "     - Tìm kiếm các cụm từ như **\"Quê quán:\", \"Nơi đăng ký khai sinh:\", \"Place of origin:\", \"Place of birth:\"** trong cả OCR mặt trước và sau.\n"
#             "     - **Trích xuất và chỉnh sửa `place_of_origin`** từ dữ liệu OCR nếu nó bị thiếu trong mã QR.\n"
#             "   - **`expiration_date`**:\n"
#             "     - Nhận diện từ các cụm từ như **\"Có giá trị đến:\", \"Date of expiry:\", \"Ngày, tháng, năm hết hạn:\"**.\n"
#             "     - **Chỉnh sửa các lỗi OCR**, như thiếu số trong năm.\n"
#             "   - **`place_of_residence`**:\n"
#             "     - Thường xuất hiện sau **\"Nơi thường trú:\", \"Place of residence:\", \"Nơi cư trú:\", \"Địa chỉ:\"**.\n"
#             "     - Có thể trải dài qua nhiều dòng; **kết hợp các dòng để tạo địa chỉ đầy đủ**.\n\n"
#             "7. **Xử lý Thông tin Nhiều dòng hoặc Bị Chia Cắt:**\n"
#             "   - Xử lý thông tin trải dài qua nhiều dòng hoặc bị chia thành các phần khác nhau.\n"
#             "   - **Kết hợp các dòng liên quan** để tái tạo đầy đủ địa chỉ hoặc các thông tin bị chia cắt.\n\n"
#             "8. **Chú ý Ngữ cảnh:**\n"
#             "   - Chú ý tới thông tin trên các dòng không mong đợi.\n"
#             "   - **Sử dụng các manh mối ngữ cảnh** để kết hợp hoặc tách thông tin đúng cách.\n\n"
#             "9. **Định dạng Kết quả:**\n"
#             "   - **Chỉ trả về một đối tượng JSON** mà không có thêm văn bản, giải thích, hoặc chú thích nào khác.\n"
#             "   - **Bao gồm tất cả các trường trong JSON**, ngay cả khi chúng trống (`\"\"`).\n\n"
#             "### Cấu trúc JSON (giá trị nên bằng tiếng Việt):\n\n"
#             "{\n"
#             '  "id_number": "",\n'
#             '  "id_number_old": "",\n'
#             '  "fullname": "",\n'
#             '  "day_of_birth": "",\n'
#             '  "sex": "",\n'
#             '  "nationality": "",\n'
#             '  "place_of_residence": "",\n'
#             '  "place_of_origin": "",\n'
#             '  "expiration_date": "",\n'
#             '  "date_of_issue": ""\n'
#             "}\n"
#         )
#         return context

#     except Exception as e:
#         logger.error(f"Lỗi tạo context người dùng: {e}")
#         raise Exception("Lỗi tạo context người dùng")
