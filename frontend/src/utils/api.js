const API_BASE_URL = "http://127.0.0.1:5000";

/**
 * Uploads an image to the Flask API for processing.
 * @param {File} image - The image file to upload.
 * @returns {Promise<Object>} - The API response containing URLs for the generated files.
 */
export const uploadImage = async (image) => {
  const formData = new FormData();
  formData.append("image", image);

  try {
    const response = await fetch(`${API_BASE_URL}/process-image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error uploading image:", error);
    throw error;
  }
};