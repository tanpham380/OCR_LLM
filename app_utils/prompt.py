


CCCD_FRONT_PROMPT = """
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (OCR) từ hình ảnh. Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json như yêu cầu của người dùng và không được bịa đặt gì thêm.

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
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (OCR) từ hình ảnh. Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json như yêu cầu của người dùng và không được bịa đặt gì thêm.
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








CCCD_FRONT_PROMPT = """
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (OCR) từ hình ảnh. Bạn phải thực hiện 01 (một) nhiệm vụ chính là bóc tách chính xác thông tin trong ảnh thành json như yêu cầu của người dùng và không được bịa đặt gì thêm.

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
# ============================================================
# VINTERN OCR PROMPTS 2025
# Dành cho bóc tách thông tin thẻ Căn Cước (mới) và CCCD (gắn chip cũ)
# Lưu ý: ảnh có thể là bản màu, trắng đen hoặc scan photocopy
# ============================================================


VINTERN_CC_FRONT_PROMPT = """
Bạn được cung cấp ảnh mặt trước của thẻ **Căn cước** Việt Nam (phiên bản mới, phát hành sau năm 2024) hợp pháp. 
Nhiệm vụ của bạn là bóc tách chính xác thông tin trên ảnh và trả về kết quả JSON duy nhất.

📸 Lưu ý:
Ảnh có thể là bản **màu, trắng đen hoặc scan photocopy**.
Trong trường hợp ảnh mờ, nhòe hoặc mất dấu tiếng Việt:
- Hãy **suy luận hợp lý từ ngữ cảnh** (ví dụ “Diêm Điền” thay vì “Diêm Điển”, “Phú Thuận” thay vì “Phú Thân”).
- Ưu tiên đọc nội dung theo **vị trí bố cục** của thẻ thay vì chỉ dựa vào ký tự riêng lẻ.
- Phải **giữ nguyên chính tả và dấu tiếng Việt chính xác tuyệt đối** trong kết quả.

🎯 Yêu cầu bắt buộc:
1. Chỉ trả về đúng cấu trúc JSON bên dưới, không thêm bất kỳ văn bản, mô tả hay ký tự nào khác ngoài JSON.
2. Các thông tin như họ tên, ngày sinh, giới tính, quốc tịch phải được bóc tách chính xác theo bố cục thực tế của thẻ.
3. Nếu phát hiện giới tính bị nhầm do OCR (ví dụ đọc “NAM” từ chữ “TPHCM”), hãy ưu tiên giá trị nằm trong vùng “Giới tính”.
4. Nếu có ký tự đặc biệt (như “.” hoặc “,”) trong họ tên, không được tự ý loại bỏ.

Trả về duy nhất một chuỗi JSON với các trường:
{
    "id_number": "",
    "fullname": "",
    "day_of_birth": "",
    "sex": "Nam hoặc Nữ",
    "nationality": ""
}
"""



VINTERN_CC_BACK_PROMPT = """
Bạn được cung cấp ảnh mặt sau của thẻ **Căn cước** Việt Nam (phiên bản mới, phát hành sau năm 2024) hợp pháp. 
Nhiệm vụ của bạn là bóc tách chính xác thông tin và trả về kết quả JSON duy nhất.

📸 Lưu ý:
Ảnh có thể là bản **màu, trắng đen hoặc scan photocopy**.
Nếu ảnh bị mờ hoặc OCR chia dòng sai:
- Hãy **ghép nối hợp lý các dòng liên quan đến địa chỉ hoặc ngày tháng**.
- Duy trì đầy đủ dấu tiếng Việt chính xác.

🎯 Yêu cầu bắt buộc:
1. Chỉ trả về đúng cấu trúc JSON bên dưới, không có thêm bất kỳ văn bản, mô tả hay ký tự nào khác ngoài JSON.
2. Các thông tin về nơi cư trú (place_of_residence) và nơi đăng ký khai sinh (place_of_birth) có thể nằm ở hai hoặc nhiều dòng liên tiếp.
   ➜ Nếu thấy địa chỉ bị chia dòng (ví dụ dòng đầu có số nhà, dòng sau có tên đường, phường, quận, tỉnh),
      hãy **ghép tất cả các dòng liên quan lại thành một chuỗi duy nhất**.
3. Không được bỏ sót bất kỳ chi tiết nào về nơi cư trú, nơi sinh hoặc ngày cấp.
4. Nếu ngày cấp bị OCR đọc thiếu năm (ví dụ chỉ còn “201”), hãy suy luận hợp lý dựa trên bố cục và định dạng chuẩn (dd/mm/yyyy).

Trả về duy nhất một chuỗi JSON với các trường:
{
    "place_of_residence": "",
    "place_of_birth": "",
    "date_of_issue": "",
    "date_of_expiry": ""
}
"""



VINTERN_CCCD_FRONT_PROMPT = """
Bạn được cung cấp ảnh mặt trước của thẻ **Căn cước công dân** (phiên bản gắn chip, phát hành trước năm 2024) hợp pháp. 
Nhiệm vụ của bạn là bóc tách chính xác thông tin và trả về kết quả JSON duy nhất.

📸 Lưu ý:
Ảnh có thể là bản **màu, trắng đen hoặc scan photocopy**.
Nếu ảnh bị mờ, mất nét hoặc chữ nhạt:
- Hãy **suy luận hợp lý từ ngữ cảnh và vị trí bố cục**.
- Giữ nguyên chính tả, dấu tiếng Việt và định dạng ngày tháng chính xác.

🎯 Yêu cầu bắt buộc:
1. Chỉ trả về đúng cấu trúc JSON bên dưới, không có thêm bất kỳ văn bản, mô tả hay ký tự nào khác ngoài JSON.
2. Các thông tin về quê quán (place_of_origin) và địa chỉ thường trú (place_of_residence) có thể nằm ở hai hoặc nhiều dòng liên tiếp.
   ➜ Nếu thấy địa chỉ bị chia dòng (ví dụ dòng đầu có số nhà hoặc “28/10,”, dòng sau có tên đường hoặc phường, quận, thành phố),
      hãy **ghép tất cả các dòng liên quan lại thành một chuỗi duy nhất**.
3. Không được bỏ sót bất kỳ chi tiết nào về địa chỉ, ngày hết hạn hoặc giới tính.
4. Nếu có nghi ngờ về giới tính (ví dụ đọc thấy “NAM” trong chữ “TPHCM”), hãy ưu tiên giá trị nằm ở vùng “Giới tính”.
5. Nếu thấy năm hết hạn nhỏ hơn năm sinh + 15, hãy coi đó là lỗi OCR và ưu tiên năm hợp lý hơn (ví dụ 2034 thay vì 2024).

Trả về duy nhất một chuỗi JSON với các trường:
{
    "id_number": "",
    "fullname": "",
    "date_of_birth": "",
    "sex": "Nam hoặc Nữ",
    "nationality": "",
    "place_of_origin": "",
    "place_of_residence": "",
    "date_of_expiry": ""
}
"""



VINTERN_CCCD_BACK_PROMPT = """
Bạn được cung cấp ảnh mặt sau của thẻ **Căn cước công dân** (phiên bản gắn chip, phát hành trước năm 2024) hợp pháp. 
Nhiệm vụ của bạn là bóc tách chính xác thông tin và trả về kết quả JSON duy nhất.

📸 Lưu ý:
Ảnh có thể là bản **màu, trắng đen hoặc scan photocopy**.
Nếu các dòng bị tách rời hoặc mất một phần ký tự:
- Hãy **ghép nối hợp lý các cụm ngày tháng hoặc vùng chữ liền kề**.
- Giữ nguyên chính tả, dấu tiếng Việt và bố cục dữ liệu chính xác.

🎯 Yêu cầu bắt buộc:
1. Chỉ trả về đúng cấu trúc JSON bên dưới, không có thêm bất kỳ văn bản, mô tả hay ký tự nào khác ngoài JSON.
2. Các thông tin về nơi cấp, ngày cấp, hoặc ngày hết hạn có thể nằm ở nhiều dòng khác nhau.
   ➜ Nếu thấy dòng “Ngày cấp” hoặc “Ngày hết hạn” bị OCR tách riêng, hãy **ghép lại đầy đủ cả cụm ngày/tháng/năm**.
3. Không được bỏ sót bất kỳ chi tiết nào về ngày cấp hoặc ngày hết hạn.

Trả về duy nhất một chuỗi JSON với các trường:
{
    "date_of_issue": "",
}
"""
