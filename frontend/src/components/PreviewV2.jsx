import { useLocation, useNavigate } from "react-router-dom";
import React, { useState, useRef, useCallback } from "react";
import {
  AlertCircle,
  ArrowLeft,
  Eye,
  Box,
  Rotate3d,
  Move3d,
  ImageIcon
} from "lucide-react";
import { Button } from "./ui/button";
import ModelViewer from "./ModelViewer";
import DownloadModel from "./DownloadModel";

const PreviewV2 = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const apiResponse = location.state?.apiResponse;
  const [isWireframe, setIsWireframe] = useState(false);
  const [transformMode, setTransformMode] = useState("translate");
  const [loadingStatus, setLoadingStatus] = useState("Loading model...");
  const [loadingError, setLoadingError] = useState(null);
  const objectRef = useRef(null); // Ref to the 3D object in ModelViewer

  // Memoize the handleTransform function
  const handleTransform = useCallback(
    (axis, amount) => {
      if (!objectRef.current) return;

      switch (transformMode) {
        case "translate":
          if (axis === "x") objectRef.current.position.x += amount;
          if (axis === "y") objectRef.current.position.y += amount;
          if (axis === "z") objectRef.current.position.z += amount;
          break;
        case "rotate":
          if (axis === "x") objectRef.current.rotation.x += amount;
          if (axis === "y") objectRef.current.rotation.y += amount;
          if (axis === "z") objectRef.current.rotation.z += amount;
          break;
        case "scale":
          const scaleAmount = amount * 0.1 + 1;
          if (axis === "x") objectRef.current.scale.x *= scaleAmount;
          if (axis === "y") objectRef.current.scale.y *= scaleAmount;
          if (axis === "z") objectRef.current.scale.z *= scaleAmount;
          if (axis === "all") {
            objectRef.current.scale.x *= scaleAmount;
            objectRef.current.scale.y *= scaleAmount;
            objectRef.current.scale.z *= scaleAmount;
          }
          break;
        default:
          break;
      }
    },
    [transformMode]
  );

  // Memoize the resetTransform function
  const resetTransform = useCallback(() => {
    if (objectRef.current) {
      objectRef.current.position.set(0, 0, 0);
      objectRef.current.rotation.set(0, 0, 0);
      objectRef.current.scale.set(1, 1, 1);
    }
  }, []);

  return (
    <div className="w-full h-screen relative overflow-hidden bg-black">
      {/* Fullscreen canvas */}
      {apiResponse && (
        <ModelViewer
          obj={apiResponse.obj}
          mtl={apiResponse.mtl}
          texture={apiResponse.texture}
          isWireframe={isWireframe}
          setObjectRef={objectRef} // Pass the ref to ModelViewer
          setLoadingStatus={setLoadingStatus}
          setLoadingError={setLoadingError}
        />
      )}

      {/* Header overlay */}
      <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center bg-black bg-opacity-50">
        <div>
          <h1 className="text-2xl font-bold text-white">3D Model Preview</h1>
          <p className="text-blue-100">
            {loadingError ? (
              <span className="text-red-300 flex items-center gap-2">
                <AlertCircle className="h-4 w-4" /> {loadingStatus}
              </span>
            ) : (
              <span>{loadingStatus}</span>
            )}
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => navigate("/")}
          className="bg-transparent border-white text-white hover:bg-white hover:text-black"
        >
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to Upload
        </Button>
      </div>

      {/* Scrollable tools panel */}
      <div className="absolute top-20 right-0 bottom-0 w-full/3 bg-gray-900 bg-opacity-90 p-4 overflow-y-auto">
        <div className="space-y-6">
          {/* View mode controls */}
          <div className="bg-gray-800 rounded-lg p-4 shadow-md">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
              <Eye className="mr-2 h-5 w-5 text-blue-400" /> View Mode
            </h3>
            <div className="grid grid-cols-2 gap-2">
              <Button
                className={`${!isWireframe
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                  } text-white`}
                onClick={() => setIsWireframe(false)}
              >
                Solid
              </Button>
              <Button
                className={`${isWireframe
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                  } text-white`}
                onClick={() => setIsWireframe(true)}
              >
                Wireframe
              </Button>
            </div>
          </div>

          {/* Transform controls */}
          <div className="bg-gray-800 rounded-lg p-4 shadow-md">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
              Transform
            </h3>
            <div className="grid grid-cols-3 gap-2 mb-4">
              <Button
                className={`${transformMode === "translate"
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                  } text-white`}
                onClick={() => setTransformMode("translate")}
              >
                <Move3d className="mr-1 h-4 w-4" /> Move
              </Button>
              <Button
                className={`${transformMode === "rotate"
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                  } text-white`}
                onClick={() => setTransformMode("rotate")}
              >
                <Rotate3d className="mr-1 h-4 w-4" /> Rotate
              </Button>
              <Button
                className={`${transformMode === "scale"
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                  } text-white`}
                onClick={() => setTransformMode("scale")}
              >
                <Box className="mr-1 h-4 w-4" /> Scale
              </Button>
            </div>

            {/* Axis controls */}
            <div className="space-y-2">
              <div className="grid grid-cols-3 gap-2">
                <Button
                  className="bg-red-700 hover:bg-red-600"
                  onClick={() => handleTransform("x", -0.1)}
                >
                  X-
                </Button>
                <Button
                  className="bg-red-600 hover:bg-red-500"
                  onClick={() => handleTransform("x", 0.1)}
                >
                  X+
                </Button>
                {transformMode === "scale" && (
                  <Button
                    className="bg-red-600 hover:bg-red-500"
                    onClick={() => handleTransform("all", 0.1)}
                  >
                    All+
                  </Button>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Button
                  className="bg-green-700 hover:bg-green-600"
                  onClick={() => handleTransform("y", -0.1)}
                >
                  Y-
                </Button>
                <Button
                  className="bg-green-600 hover:bg-green-500"
                  onClick={() => handleTransform("y", 0.1)}
                >
                  Y+
                </Button>
                {transformMode === "scale" && (
                  <Button
                    className="bg-red-700 hover:bg-red-600"
                    onClick={() => handleTransform("all", -0.1)}
                  >
                    All-
                  </Button>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  className="bg-blue-700 hover:bg-blue-600"
                  onClick={() => handleTransform("z", -0.1)}
                >
                  Z-
                </Button>
                <Button
                  className="bg-blue-600 hover:bg-blue-500"
                  onClick={() => handleTransform("z", 0.1)}
                >
                  Z+
                </Button>
              </div>
            </div>

            {apiResponse && (
              <DownloadModel obj={apiResponse.obj} mtl={apiResponse.mtl} tex={apiResponse.texture}/>
            )}

            {/* Reset button */}
            <Button
              className="w-full mt-4 bg-yellow-600 hover:bg-yellow-700 text-white"
              onClick={resetTransform}
            >
              Reset Transform
            </Button>
          </div>

          {/* Reference image */}
          {/* {image && (
            <div className="bg-gray-800 rounded-lg p-4 shadow-md">
              <h3 className="text-lg font-semibold text-white mb-2 flex items-center">
                <ImageIcon className="mr-2 h-5 w-5 text-blue-400" /> Reference
                Image
              </h3>
              <div className="border border-gray-700 rounded-lg overflow-hidden">
                <img
                  src={image}
                  alt="Reference"
                  className="w-full h-48 object-contain"
                />
              </div>
            </div>
          )} */}
        </div>
      </div>
    </div>
  );
};

export default PreviewV2;