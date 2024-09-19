from quart import Blueprint, request, jsonify
from app_utils.file_handler import secure_file
from app_utils.logging import get_logger
from app_utils.middleware import require_api_key
from app_utils.service import process_with_llm_custom, scan

blueprint = Blueprint("routes", __name__)
logger = get_logger(__name__)

@blueprint.route("/")
async def home():
    """
    Home route to check if the API is running.
    """
    version = "Hello, World!"
    return jsonify({"successful": True, "message": version, "data": None}), 200

@blueprint.route("/api/v1/scan_cccd", methods=['POST'])
@require_api_key
async def scan_cccd():
    """
    API endpoint to scan ID card images (CCCD).
    """
    try:
        file = (await request.files).get("image")
        file2 = (await request.files).get("image2")
        if not file or not file2:
            return jsonify({"successful": False, "message": "No image found in the request", "data": None}), 400

        
        if file and file2:
            list_path = [secure_file(file) , secure_file(file2)] 
            for i in list_path :
                if "not allowed" in i:
                    return jsonify({"successful": False, "message": "File extension not allowed", "data": None}), 400
            llm_response = await scan(list_path)
            return jsonify({"successful": True, "message": "OCR performed successfully", "data": llm_response}), 200
        else:
            return jsonify({"successful": False, "message": "No image found in the request", "data": None}), 400
    except Exception as e:
        return jsonify({"successful": False, "message": str(e), "data": None}), 500


@blueprint.route("/api/v1/process_llm_custom", methods=['POST'])
@require_api_key
async def process_llm_custom_route():
    try:
        data = await request.json
        system_prompt = data.get("system_prompt", "")
        user_prompt = data.get("user_prompt")
        custom_image = data.get("custom_image", "")

        if user_prompt is None:
            return jsonify({"result": False, "message": "User prompt cannot be None."}), 400
        
        # Call the service function
        llm_custom_result = await process_with_llm_custom(system_prompt, user_prompt, custom_image)
        
        return jsonify({"result": True, "data": llm_custom_result, "message": "LLM processing successful."}), 200

    except ValueError as ve:
        return jsonify({"result": False, "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"result": False, "message": str(e)}), 500
