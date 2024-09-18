from functools import wraps
from quart import request, jsonify

from config import API_KEY


def require_api_key(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == API_KEY:
            return await f(*args, **kwargs)
        else:
            return jsonify({"message": "Invalid or missing API key", "successful": False, "data": None}), 403

    return decorated_function
