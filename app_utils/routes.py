import base64
import io
import json
import os
import cv2
import numpy as np
from quart import Blueprint, request, jsonify, g, url_for
import torch
from app_utils.file_handler import save_image, secure_file
from app_utils.logging import get_logger
from app_utils.middleware import require_api_key
from app_utils.service import process_with_llm_custom, scan
from config import SAVE_IMAGES
from PIL import Image

blueprint = Blueprint("routes", __name__)
logger = get_logger(__name__)

@blueprint.route("/")
async def home():
    """
    Home route to check if the API is running.
    """
    version = "Hello, World!"
    return jsonify({"successful": True, "message": version, "data": None}), 200
@blueprint.route("/api/v1/ocr_results", methods=['GET'])
@require_api_key
async def get_ocr_results():
    """
    API endpoint to retrieve all OCR results.
    """
    try:
        db_manager = g.db_manager
        conn = await db_manager.get_connection()
        cursor = await conn.execute("SELECT * FROM ocr_results")
        ocr_results = await cursor.fetchall()
        
        result_list = [
            {
                "id": row["id"],
                # "front_side_ocr": row["front_side_ocr"],
                # "front_side_qr": row["front_side_qr"],
                # "back_side_ocr": row["back_side_ocr"],
                # "back_side_qr": row["back_side_qr"],
                "timestamp": row["timestamp"]
            }
            for row in ocr_results
        ]
        
        return jsonify({"successful": True, "message": "OCR results retrieved", "data": result_list}), 200

    except Exception as e:
        logger.error(f"Error retrieving OCR results: {e}")
        return jsonify({"successful": False, "message": str(e), "data": None}), 500

@blueprint.route("/api/v1/ocr_full_results/<int:ocr_result_id>", methods=['GET'])
@require_api_key
async def get_ocr_full_results(ocr_result_id):
    """
    API endpoint để lấy đầy đủ thông tin từ bảng ocr_results, user_contexts, và images.
    """
    try:
        db_manager = g.db_manager
        conn = await db_manager.get_connection()

        query = '''
        SELECT 
            ocr_results.id AS ocr_result_id,
            ocr_results.front_side_ocr,
            ocr_results.front_side_qr,
            ocr_results.back_side_ocr,
            ocr_results.back_side_qr,
            ocr_results.timestamp AS ocr_timestamp,
            user_contexts.context,
            user_contexts.timestamp AS context_timestamp,
            images.image_type,
            images.image_data,
            images.timestamp AS image_timestamp

        FROM 
            ocr_results
        LEFT JOIN 
            user_contexts ON ocr_results.id = user_contexts.ocr_result_id
        LEFT JOIN 
            images ON ocr_results.id = images.ocr_result_id
        WHERE 
            ocr_results.id = ?
        '''

        cursor = await conn.execute(query, (ocr_result_id,))
        result = await cursor.fetchall()

        result_list = []
        for row in result:
            # Convert base64 image to file and generate a URL for the user
            if row["image_data"]:
                image_data = base64.b64decode(row["image_data"])
                im_arr = np.frombuffer(image_data, dtype=np.uint8)  # im_arr is one-dim Numpy array

                img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

                
                image_path = save_image(img, SAVE_IMAGES)

                image_url = url_for('static', filename=f'images/{os.path.basename(image_path)}', _external=True)
            else:
                image_url = None

            result_list.append({
                "ocr_result_id": row["ocr_result_id"],
                "front_side_ocr": row["front_side_ocr"],
                "front_side_qr": row["front_side_qr"],
                "back_side_ocr": row["back_side_ocr"],
                "back_side_qr": row["back_side_qr"],
                "ocr_timestamp": row["ocr_timestamp"],
                "context": row["context"],
                "context_timestamp": row["context_timestamp"],
                "image_type": row["image_type"],
                "image_url": image_url,  # Return the URL for the image
                "image_timestamp": row["image_timestamp"]
            })

        return jsonify({"successful": True, "message": "Full OCR results retrieved", "data": result_list}), 200

    except Exception as e:
        logger.error(f"Error retrieving full OCR results: {e}")
        return jsonify({"successful": False, "message": str(e), "data": None}), 500

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
