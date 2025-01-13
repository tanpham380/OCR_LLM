


from controller.openapi_vison import Llm_Vision_Exes


ocr_controller = Llm_Vision_Exes(
    api_key="1234", 
api_base="http://192.168.1.136:2242/v1")




print(ocr_controller.generate_multi(["2024_01_22_10_57_27_resize.jpg"]))
