import re

def convert_text(text):
    conversion_map = {
        '廙': 'ư',
        'ờｯ': 'ữ',
        'ạｻ': 'ội',
        '冓': 'i',
        'ăｴ': 'ô',
        'ờｳ': 'ỳ',
        '静ｴ': 'ố',
        'Đｩ': 'ĩ',
        '蘯｣': 'ả',
        'ờ弓': 'ị',
        'Trng': 'Trương',
        'Qu': 'Quỳ',
        'nh': 'nh',
        'Nh': 'Như'
    }
    
    for encoded, decoded in conversion_map.items():
        text = text.replace(encoded, decoded)
    
    # Thêm dấu | cuối cùng nếu chưa có
    if not text.endswith('|'):
        text = text.rsplit('|', 1)
        text = '|'.join(text[:-1]) + '|' + text[-1]
    
    # Sửa lỗi "Độii" thành "Đội"
    text = text.replace('Độii', 'Đội')
    
    return text

# Ví dụ sử dụng
encoded_text = "045306004945||Trng Qu廙軟h Nh|02052006|Nờｯ|Đạｻ冓 9, Thăｴn Huờｳnh Căｴng Đ静ｴng, Trung Nam, VĐｩnh Linh, Qu蘯｣ng Trờ弓12102022"
decoded_text = convert_text(encoded_text)
print(decoded_text)
