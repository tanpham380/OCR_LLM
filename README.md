OCR with LLM
This project is an API service for Optical Character Recognition (OCR) on ID cards, enhanced with Large Language Model (LLM) post-processing for improved accuracy and data extraction.

Key Features:

ID Card Detection: Accurately detects and extracts ID cards from images.
QR Code Reading: Extracts information from QR codes present on ID cards.
OCR Engine: Utilizes a robust OCR engine (currently unspecified in provided code) for text extraction.
LLM Integration: Employs an LLM (likely Qwen2.5 based on code) to process and correct OCR output, ensuring higher accuracy and structured data extraction.
API Endpoint: Provides a /api/v1/scan_cccd endpoint for easy integration with other applications.
Security: Implements API key authentication for secure access.
Performance Optimization: Leverages asynchronous processing (asyncio) for concurrent image handling and faster response times.
Logging: Includes detailed logging for monitoring and debugging.
Project Structure:

app.py: Initializes the Flask application, sets up logging, and registers blueprints for routing.
app_utils: Contains utility modules for file handling, logging, middleware, database interaction, and service logic.
file_handler.py: Provides functions for secure file saving, image processing, and format conversion.
logging.py: Sets up custom logging to separate log files by level (info, debug, warning, error).
middleware.py: Implements API key authentication middleware.
database.py: Manages SQLite database connections and operations.
service.py: Contains the core logic for image processing, OCR, QR code reading, and LLM interaction.
controller: Likely houses controllers for interacting with the detector and LLM models.
detecter_controller.py: (Assumed) Handles image detection, QR code reading, and orientation correction.
llm_controller.py: (Assumed) Manages communication with the LLM, including context setting and response parsing.
config.py: Stores configuration settings, including API keys, model paths, and thresholds.
main.py: Creates the Flask app and runs it using Uvicorn.
test.py: A Gradio demo showcasing image rotation capabilities using a Vintern model.
Getting Started:

Installation:
Clone the repository.
Install required dependencies: pip install -r requirements.txt
Configure the .env file with your API keys and other settings.
Running the API:
Start the Uvicorn server: uvicorn main:scan_app --reload --host 0.0.0.0 --port 5001
API Usage:
Send a POST request to /api/v1/scan_cccd with two image files (front and back of ID card) using the image and image2 keys in the request body.
Include your API key in the X-API-Key header.
Future Improvements:

Documentation: Add more detailed API documentation, including request/response examples and error codes.
Model Specifics: Provide more information about the chosen OCR engine and LLM model for better understanding and potential customization.
Error Handling: Implement more robust error handling and return informative error messages to the API client.
Testing: Develop comprehensive unit and integration tests to ensure code quality and functionality.
Contributing:

Contributions to this project are welcome. Please fork the repository and submit pull requests for any bug fixes, improvements, or new features.