from flask import Flask, request, jsonify, send_file
import os
from backend.voxel_processing.main import process_image  # Import the refactored function

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        # Call the process_image function to process the image
        obj_path, mtl_path, texture_path = process_image(image_path, OUTPUT_FOLDER)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the generated files
    return jsonify({
        'obj': f'/download/{os.path.basename(obj_path)}',
        'mtl': f'/download/{os.path.basename(mtl_path)}',
        'texture': f'/download/{os.path.basename(texture_path)}'
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)