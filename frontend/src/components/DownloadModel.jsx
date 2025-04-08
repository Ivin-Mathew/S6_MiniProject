import React from "react";
import { Button } from "./ui/button";
import { Download } from "lucide-react";

const DownloadModel = ({ obj, mtl, tex }) => {
  const handleDownloadModel = () => {
    if (!obj || !mtl || !tex) {
      console.error("No model available to download");
      return;
    }

    // Function to trigger download using fetch to force download
    const downloadFile = async (url, filename) => {
      try {
        console.log(`Attempting to download ${url} as ${filename}`);
        
        // Use fetch to get the file content as a blob
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        
        const blob = await response.blob();
        
        // Create object URL for the blob
        const objectUrl = window.URL.createObjectURL(blob);
        
        // Create an anchor element with download attribute
        const link = document.createElement("a");
        link.href = objectUrl;
        link.download = filename; // This forces download instead of navigation
        link.style.display = "none";
        
        // Add to DOM, click it, and remove it
        document.body.appendChild(link);
        link.click();
        
        // Clean up
        setTimeout(() => {
          document.body.removeChild(link);
          window.URL.revokeObjectURL(objectUrl);
        }, 100);
        
        console.log(`Download initiated for ${filename}`);
      } catch (error) {
        console.error(`Error downloading ${filename}:`, error);
      }
    };

    // Get the filenames from the paths
    const objFilename = obj.split("/").pop();
    const mtlFilename = mtl.split("/").pop();
    const texFilename = tex.split("/").pop();

    // Initiate downloads
    downloadFile(obj, objFilename || "model.obj");
    downloadFile(mtl, mtlFilename || "model.mtl");
    downloadFile(tex, texFilename || "model.png");

    console.log(
      `Downloading model: ${objFilename}, materials: ${mtlFilename}, texture: ${texFilename}`
    );
  };

  return (
    <Button
      className="w-full mt-4 bg-green-600 hover:bg-green-700 text-white"
      onClick={handleDownloadModel}
    >
      <Download className="mr-2 h-5 w-5" /> Download OBJ, MTL and Texture Files
    </Button>
  );
};

export default DownloadModel;