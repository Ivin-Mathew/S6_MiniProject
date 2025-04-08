from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from .voxel_processing.main import process_image  # Import the refactored function

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return """
    <h1>Welcome to the Voxel Generation API</h1>
    <p>This API allows you to upload an image and generate 3D model files (.obj, .mtl, and texture).</p>
    <h2>Endpoints:</h2>
    <ul>
        <li>
            <strong>POST /process-image</strong><br>
            Upload an image to generate 3D model files.<br>
            <strong>Request:</strong> Form-data with a key <code>image</code> containing the image file.<br>
            <strong>Response:</strong> JSON with URLs to download the generated files.
        </li>
        <li>
            <strong>GET /download/&lt;filename&gt;</strong><br>
            Download a specific generated file.<br>
            <strong>Request:</strong> Provide the filename in the URL.<br>
            <strong>Response:</strong> The requested file.
        </li>
    </ul>
    <h2>Example Usage:</h2>
    <p><strong>POST /process-image</strong></p>
    <pre>
    curl -X POST -F "image=@path/to/image.jpg" http://127.0.0.1:5000/process-image
    </pre>
    <p>The response will include URLs to download the generated .obj, .mtl, and texture files.</p>
    """

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

    # Construct the full URLs for the generated files
    base_url = request.host_url.rstrip('/')  # Get the base URL of the backend
    return jsonify({
        'obj': f'{base_url}/download/{os.path.basename(obj_path)}',
        'mtl': f'{base_url}/download/{os.path.basename(mtl_path)}',
        'texture': f'{base_url}/download/{os.path.basename(texture_path)}'
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)