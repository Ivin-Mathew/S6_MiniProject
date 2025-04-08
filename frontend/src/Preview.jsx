import { useLocation, useNavigate } from "react-router-dom";
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { TextureLoader } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Button } from "./components/ui/button";
import {
  AlertCircle,
  ArrowLeft,
  Eye,
  Box,
  Rotate3d,
  Move3d,
  ImageIcon,
  Download
} from "lucide-react";

const Preview = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const image = location.state?.apiResponse;
  const canvasRef = useRef(null);
  const [isWireframe, setIsWireframe] = useState(false);
  const [transformMode, setTransformMode] = useState("translate"); // 'translate', 'rotate', 'scale'
  const [currentObject, setCurrentObject] = useState("assets/output/mesh.obj");
  const [currentMaterial, setCurrentMaterial] = useState("assets/output/mesh.mtl");
  const [loadingStatus, setLoadingStatus] = useState("Loading model...");
  const [loadingError, setLoadingError] = useState(null);

  // Store scene objects reference
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const objectRef = useRef(null);
  const materialsRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true
    });
    renderer.setSize(
      canvasRef.current.clientWidth,
      canvasRef.current.clientHeight
    );
    rendererRef.current = renderer;

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(20, 20);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    const mtlLoader = new MTLLoader();
    // Set the texture path for the MTL loader
    

    mtlLoader.load(
      currentMaterial, // Just use the filename, since we set the path above
      function (materials) {
        materials.preload();
        materialsRef.current = materials;

        // Explicitly load and apply the texture - important addition
        const textureLoader = new THREE.TextureLoader();
        textureLoader.setPath("/");
        const texture = textureLoader.load(
          "mesh.png",
          (loadedTexture) => {
            console.log("Texture loaded successfully:", loadedTexture);
            // Apply the loaded texture to all materials that need it
            Object.values(materials.materials).forEach((material) => {
              material.map = loadedTexture;
              material.needsUpdate = true;
            });

            // Force update if the object is already loaded
            if (objectRef.current) {
              objectRef.current.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                  if (Array.isArray(child.material)) {
                    child.material.forEach((mat) => {
                      mat.map = loadedTexture;
                      mat.needsUpdate = true;
                    });
                  } else if (child.material) {
                    child.material.map = loadedTexture;
                    child.material.needsUpdate = true;
                  }
                }
              });
            }
          },
          undefined,
          (error) => console.error("Error loading texture:", error)
        );

        // After loading materials, load the OBJ file
        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);
        // objLoader.setPath("../assets/"); // Set the path for OBJ loader too

        objLoader.load(
          currentObject, // Just use the filename
          function (object) {
            // Success callback - rest of your code remains the same
            object.traverse(function (child) {
              if (child instanceof THREE.Mesh) {
                // If the material didn't load properly, create one with the texture
                if (
                  !child.material ||
                  (Array.isArray(child.material) && child.material.length === 0)
                ) {
                  // Try to get the texture we loaded earlier
                  const meshTexture = textureLoader.load("mesh.png");

                  child.material = new THREE.MeshStandardMaterial({
                    color: 0xffffff, // White color to not tint the texture
                    wireframe: isWireframe,
                    map: meshTexture // Apply texture directly
                  });
                } else {
                  // Handle existing materials
                  if (Array.isArray(child.material)) {
                    child.material.forEach((mat) => {
                      mat.wireframe = isWireframe;

                      // Ensure the texture is applied to each material
                      if (!mat.map) {
                        mat.map = textureLoader.load("mesh.png");
                      }

                      // Optimize texture settings
                      if (mat.map) {
                        mat.map.anisotropy = 16;
                        mat.needsUpdate = true;
                      }
                    });
                  } else {
                    child.material.wireframe = isWireframe;

                    // Ensure the texture is applied to the material
                    if (!child.material.map) {
                      child.material.map = textureLoader.load("mesh.png");
                    }

                    // Optimize texture settings
                    if (child.material.map) {
                      child.material.map.anisotropy = 16;
                      child.material.needsUpdate = true;
                    }
                  }
                }
              }
            });

            scene.add(object);
            objectRef.current = object;
            setLoadingStatus("Model loaded successfully!");
            setLoadingError(null);
          },
          // Loading progress callback remains the same
          function (xhr) {
            if (xhr.lengthComputable) {
              const percentComplete = (xhr.loaded / xhr.total) * 100;
              setLoadingStatus(`Loading: ${Math.round(percentComplete)}%`);
            }
          },
          // Error callback remains the same
          function (error) {
            console.error(
              `Failed to load OBJ model from ${currentObject}`,
              error
            );
            setLoadingError(
              "Failed to load model. Please check the file path and ensure the file exists."
            );
            setLoadingStatus("Error loading model");

            // Add a default cube as a placeholder
            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshStandardMaterial({
              color: 0xff0000,
              wireframe: isWireframe
            });
            const cube = new THREE.Mesh(geometry, material);
            scene.add(cube);
            objectRef.current = cube;
          }
        );
      },
      // MTL Loading progress callback remains the same
      function (xhr) {
        if (xhr.lengthComputable) {
          const percentComplete = (xhr.loaded / xhr.total) * 100;
          setLoadingStatus(
            `Loading materials: ${Math.round(percentComplete)}%`
          );
        }
      },
      // MTL Loading error callback remains the same
      function (error) {
        console.error(
          `Failed to load MTL material from ${currentMaterial}`,
          error
        );
        setLoadingStatus(
          "Error loading materials. Continuing with default materials."
        );

        // Continue loading the OBJ without materials
        loadObjectWithoutMaterials();
      }
    );

    // Function to load object if material loading fails
    const loadObjectWithoutMaterials = () => {
      const objLoader = new OBJLoader();

      objLoader.load(
        currentObject,
        function (object) {
          // Apply default material
          object.traverse(function (child) {
            if (child instanceof THREE.Mesh) {
              child.material = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                wireframe: isWireframe
              });
            }
          });

          scene.add(object);
          objectRef.current = object;
          setLoadingStatus("Model loaded successfully (without materials)!");
          setLoadingError(null);
        },
        // Loading progress
        function (xhr) {
          if (xhr.lengthComputable) {
            const percentComplete = (xhr.loaded / xhr.total) * 100;
            setLoadingStatus(`Loading: ${Math.round(percentComplete)}%`);
          }
        },
        // Loading error
        function (error) {
          console.error(`Failed to load model from ${currentObject}`, error);
          setLoadingStatus("Error loading model");
          setLoadingError(
            "Failed to load model. Please check the file path and ensure the file exists."
          );

          // Add a default cube as a placeholder
          const geometry = new THREE.BoxGeometry(1, 1, 1);
          const material = new THREE.MeshStandardMaterial({
            color: 0xff0000,
            wireframe: isWireframe
          });
          const cube = new THREE.Mesh(geometry, material);
          scene.add(cube);
          objectRef.current = cube;
        }
      );
    };

    // Handle window resize
    const handleResize = () => {
      if (canvasRef.current) {
        const width = canvasRef.current.clientWidth;
        const height = canvasRef.current.clientHeight;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        renderer.setSize(width, height);
      }
    };

    window.addEventListener("resize", handleResize);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      if (controlsRef.current) {
        controlsRef.current.update();
      }

      renderer.render(scene, camera);
    };

    animate();

    // Clean up
    return () => {
      window.removeEventListener("resize", handleResize);
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // Update wireframe mode when the state changes
  useEffect(() => {
    if (objectRef.current) {
      objectRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          if (Array.isArray(child.material)) {
            child.material.forEach((mat) => {
              mat.wireframe = isWireframe;
            });
          } else if (child.material) {
            child.material.wireframe = isWireframe;
          }
        }
      });
    }
  }, [isWireframe]);

  // Handle transform mode changes
  useEffect(() => {
    if (controlsRef.current) {
      // In a real implementation, you would use TransformControls instead of OrbitControls
      // for moving, rotating, and scaling. This is a simplified representation.
      console.log(`Transform mode changed to: ${transformMode}`);
    }
  }, [transformMode]);

  // Event handlers for model transformation
  const handleTransform = (axis, amount) => {
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
  };

  const handleDownloadModel = () => {
    if (!objectRef.current) {
      console.error("No model available to download");
      return;
    }

    // Create download links for both OBJ and MTL files

    // Get the filenames from the paths
    const objFilename = currentObject.split("/").pop();
    const mtlFilename = currentMaterial.split("/").pop();

    // Download OBJ file
    const objLink = document.createElement("a");
    objLink.href = currentObject;
    objLink.download = objFilename || "model.obj";
    document.body.appendChild(objLink);
    objLink.click();
    document.body.removeChild(objLink);

    // Download MTL file
    const mtlLink = document.createElement("a");
    mtlLink.href = currentMaterial;
    mtlLink.download = mtlFilename || "model.mtl";
    document.body.appendChild(mtlLink);
    mtlLink.click();
    document.body.removeChild(mtlLink);

    console.log(
      `Downloading model: ${objFilename} and materials: ${mtlFilename}`
    );
  };

  return (
    <div className="w-full h-screen relative overflow-hidden bg-black">
      {/* Fullscreen canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full absolute top-0 left-0"
      ></canvas>

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
                className={`${
                  !isWireframe
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                } text-white`}
                onClick={() => setIsWireframe(false)}
              >
                Solid
              </Button>
              <Button
                className={`${
                  isWireframe
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
                className={`${
                  transformMode === "translate"
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                } text-white`}
                onClick={() => setTransformMode("translate")}
              >
                <Move3d className="mr-1 h-4 w-4" /> Move
              </Button>
              <Button
                className={`${
                  transformMode === "rotate"
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gray-700 hover:bg-gray-600"
                } text-white`}
                onClick={() => setTransformMode("rotate")}
              >
                <Rotate3d className="mr-1 h-4 w-4" /> Rotate
              </Button>
              <Button
                className={`${
                  transformMode === "scale"
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

            <Button
              className="w-full mt-4 bg-green-600 hover:bg-green-700 text-white"
              onClick={handleDownloadModel}
            >
              <Download className="mr-2 h-5 w-5" /> Download OBJ & MTL Files
            </Button>

            {/* Reset button */}
            <Button
              className="w-full mt-4 bg-yellow-600 hover:bg-yellow-700 text-white"
              onClick={() => {
                if (objectRef.current) {
                  objectRef.current.position.set(0, 0, 0);
                  objectRef.current.rotation.set(0, 0, 0);
                  objectRef.current.scale.set(1, 1, 1);
                }
              }}
            >
              Reset Transform
            </Button>
          </div>

          {/* Reference image */}
          {image && (
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
          )}
        </div>
      </div>
    </div>
  );
};

export default Preview;