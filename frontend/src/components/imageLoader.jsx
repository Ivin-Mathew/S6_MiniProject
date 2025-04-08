import { useState } from "react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";

import { X } from "lucide-react";

const ImageUploader = ({ setImage }) => {
  const [localImage, setLocalImage] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      setLocalImage(URL.createObjectURL(file)); // Use URL.createObjectURL for preview
      setImage(file); // Pass the file object to the parent component
    }
  };

  const removeImage = () => {
    setLocalImage(null);
    setImage(null);
  };

  return (
    <Card className="w-80 border-dashed border-2 border-gray-400 flex flex-col items-center justify-center p-4">
      <CardContent className="flex flex-col items-center w-full">
        {!localImage ? (
          <label className="cursor-pointer text-blue-500 hover:underline">
            Upload an Image
            <input type="file" accept="image/*" className="hidden" onChange={handleImageChange} />
          </label>
        ) : (
          <div className="relative w-full">
            <img src={localImage} alt="Uploaded" className="w-full h-auto rounded-lg" />
            <Button
              onClick={removeImage}
              variant="destructive"
              size="icon"
              className="absolute top-2 right-2"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ImageUploader;