# Create the Flask application
import argparse
from app import create_app


scan_app = create_app()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, default=5001, help="Port of serving API"
    )
    args = parser.parse_args()
    scan_app.run(host="0.0.0.0", port=args.port)
    
      
    # (ocr2) (base) gitlab@AIMACHINE:~/ocr$ uvicorn main:scan_app --reload --host 0.0.0.0 --port 5001