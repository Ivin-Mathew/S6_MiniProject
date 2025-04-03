import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Progress } from "./components/ui/progress";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Upload, Image as ImageIcon, ArrowRight } from "lucide-react";
import ImageUploader from "./components/imageLoader";

function App() {
  const [image, setImage] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  const handleSubmit = () => {
    if (image) {
      // Simulate upload progress
      setUploading(true);
      let progress = 0;
      const interval = setInterval(() => {
        progress += 5;
        setUploadProgress(progress);
        if (progress >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            navigate("/preview", { state: { image } });
          }, 500);
        }
      }, 100);
    }
  };

  return (
    
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-black to-gray-600 p-4">
      <Card className="w-full max-w-lg bg-transparent" >
        <CardHeader className="bg-gradient-to-r  rounded-t-lg text-white">
          <div className="flex items-center gap-2">
            <div>
              <CardTitle className="text-2xl font-bold">2D to 3D Reconstruction</CardTitle>
              <CardDescription className="text-blue-100">Transform your images into 3D models</CardDescription>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="p-6 space-y-6">
          <div className="bg-blue-50 border border-blue-100 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <ImageIcon className="h-5 w-5 text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-800">
                Upload a clear image with good lighting for the best 3D reconstruction results.
              </div>
            </div>
          </div>
          
          <div className="border-2 border-dashed border-blue-200 rounded-xl p-8 flex items-center justify-center">
            <ImageUploader setImage={setImage} />
          </div>
          
          {image && (
            <Alert className="bg-green-50 border-green-200">
              <AlertDescription className="text-green-800 flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500"></div>
                Image successfully loaded and ready for processing
              </AlertDescription>
            </Alert>
          )}
          
          {uploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-500">
                <span>Preparing image for reconstruction...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}
        </CardContent>
        
        <CardFooter className="flex justify-between p-6 pt-0">
          <Button variant="outline" disabled={uploading}>
            Cancel
          </Button>
          <Button 
            className="bg-blue-600 hover:bg-blue-700 gap-1"
            onClick={handleSubmit}
            disabled={!image || uploading}
          >
            Process Image <ArrowRight className="h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}

export default App;