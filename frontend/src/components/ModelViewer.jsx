import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ModelViewer = ({
  obj,
  mtl,
  texture,
  isWireframe,
  setObjectRef,
  setLoadingStatus,
  setLoadingError
}) => {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const objectRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !obj || !mtl || !texture) {
      console.log("Missing required props:", { obj, mtl, texture });
      return;
    }

    console.log("Loading 3D model with:", { obj, mtl, texture });

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
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
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    const gridHelper = new THREE.GridHelper(20, 20);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    setLoadingStatus("Loading texture...");
    
    // First, load the texture
    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(
      texture,
      (loadedTexture) => {
        // Configure texture
        loadedTexture.flipY = false;
        loadedTexture.colorSpace = THREE.SRGBColorSpace;
        loadedTexture.encoding = THREE.sRGBEncoding;
        
        console.log("Texture loaded successfully:", texture);
        setLoadingStatus("Loading materials...");
        
        // Then load materials
        const mtlLoader = new MTLLoader();
        mtlLoader.load(
          mtl,
          (materials) => {
            materials.preload();
            
            // Manually apply the texture to each material
            Object.values(materials.materials).forEach((material) => {
              material.map = loadedTexture;
              material.needsUpdate = true;
            });
            
            console.log("Materials loaded successfully:", mtl);
            setLoadingStatus("Loading 3D model...");
            
            // Finally load the object
            const objLoader = new OBJLoader();
            objLoader.setMaterials(materials);
            
            objLoader.load(
              obj,
              (object) => {
                // Process loaded object
                object.traverse((child) => {
                  if (child instanceof THREE.Mesh) {
                    // Set wireframe mode
                    if (Array.isArray(child.material)) {
                      child.material.forEach(mat => {
                        mat.wireframe = isWireframe;
                        
                        // Ensure texture is applied
                        if (!mat.map) {
                          mat.map = loadedTexture;
                        }
                        
                        // Configure material for proper rendering
                        mat.side = THREE.DoubleSide; // Render both sides
                        mat.transparent = true;
                        mat.needsUpdate = true;
                      });
                    } else {
                      child.material.wireframe = isWireframe;
                      
                      // Ensure texture is applied
                      if (!child.material.map) {
                        child.material.map = loadedTexture;
                      }
                      
                      // Configure material for proper rendering
                      child.material.side = THREE.DoubleSide;
                      child.material.transparent = true;
                      child.material.needsUpdate = true;
                    }
                  }
                });
                
                // Position and scale the object optimally
                const box = new THREE.Box3().setFromObject(object);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                
                // Center the object
                object.position.sub(center);
                
                // Scale it to a reasonable size
                const maxDim = Math.max(size.x, size.y, size.z);
                if (maxDim > 2) {
                  const scale = 2 / maxDim;
                  object.scale.set(scale, scale, scale);
                }
                
                scene.add(object);
                objectRef.current = object;
                
                // Pass the object reference to the parent component
                if (setObjectRef && setObjectRef.current !== undefined) {
                  setObjectRef.current = object;
                }
                
                setLoadingStatus("Model loaded successfully!");
                setLoadingError(null);
              },
              // Progress callback
              (xhr) => {
                const progress = xhr.loaded / xhr.total * 100;
                setLoadingStatus(`Loading model: ${Math.round(progress)}%`);
              },
              // Error callback
              (error) => {
                console.error("Error loading OBJ:", error);
                setLoadingError(`Failed to load 3D model: ${error.message}`);
              }
            );
          },
          // MTL progress callback
          (xhr) => {
            const progress = xhr.loaded / xhr.total * 100;
            setLoadingStatus(`Loading materials: ${Math.round(progress)}%`);
          },
          // MTL error callback
          (error) => {
            console.error("Error loading MTL:", error);
            setLoadingError(`Failed to load materials: ${error.message}`);
            
            // Try loading the model without materials
            loadModelWithoutMaterials(obj, loadedTexture);
          }
        );
      },
      // Texture progress callback
      (xhr) => {
        if (xhr.lengthComputable) {
          const progress = xhr.loaded / xhr.total * 100;
          setLoadingStatus(`Loading texture: ${Math.round(progress)}%`);
        }
      },
      // Texture error callback
      (error) => {
        console.error("Error loading texture:", error);
        setLoadingError(`Failed to load texture: ${error.message}`);
        
        // Try loading without the texture
        loadModelWithoutTexture(obj, mtl);
      }
    );

    // Function to load model without materials but with texture
    const loadModelWithoutMaterials = (objUrl, loadedTexture) => {
      setLoadingStatus("Trying to load model without materials...");
      
      const objLoader = new OBJLoader();
      objLoader.load(
        objUrl,
        (object) => {
          // Create a new material with the texture
          const material = new THREE.MeshStandardMaterial({
            map: loadedTexture,
            wireframe: isWireframe,
            side: THREE.DoubleSide,
            transparent: true
          });
          
          // Apply the material to all meshes
          object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
              child.material = material;
            }
          });
          
          scene.add(object);
          objectRef.current = object;
          
          // Pass the object reference to the parent component
          if (setObjectRef && setObjectRef.current !== undefined) {
            setObjectRef.current = object;
          }
          
          setLoadingStatus("Model loaded successfully (without materials)!");
          setLoadingError(null);
        },
        // Progress callback
        (xhr) => {
          const progress = xhr.loaded / xhr.total * 100;
          setLoadingStatus(`Loading model: ${Math.round(progress)}%`);
        },
        // Error callback
        (error) => {
          console.error("Error loading OBJ without materials:", error);
          setLoadingError("Failed to load model. Using fallback cube.");
          
          // Create a default cube as fallback
          createFallbackCube(loadedTexture);
        }
      );
    };
    
    // Function to load model with materials but without texture
    const loadModelWithoutTexture = (objUrl, mtlUrl) => {
      setLoadingStatus("Trying to load model without texture...");
      
      const mtlLoader = new MTLLoader();
      mtlLoader.load(
        mtlUrl,
        (materials) => {
          materials.preload();
          
          const objLoader = new OBJLoader();
          objLoader.setMaterials(materials);
          
          objLoader.load(
            objUrl,
            (object) => {
              // Set wireframe mode
              object.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                  if (Array.isArray(child.material)) {
                    child.material.forEach(mat => {
                      mat.wireframe = isWireframe;
                    });
                  } else {
                    child.material.wireframe = isWireframe;
                  }
                }
              });
              
              scene.add(object);
              objectRef.current = object;
              
              // Pass the object reference to the parent component
              if (setObjectRef && setObjectRef.current !== undefined) {
                setObjectRef.current = object;
              }
              
              setLoadingStatus("Model loaded successfully (without texture)!");
              setLoadingError(null);
            },
            null,
            (error) => {
              console.error("Error loading model without texture:", error);
              setLoadingError("Failed to load model. Using fallback cube.");
              
              // Create a default colored cube
              createFallbackCube();
            }
          );
        },
        null,
        (error) => {
          console.error("Error loading materials without texture:", error);
          setLoadingError("Failed to load materials and texture. Using fallback cube.");
          
          // Create a default colored cube
          createFallbackCube();
        }
      );
    };
    
    // Function to create a fallback cube
    const createFallbackCube = (texture = null) => {
      const geometry = new THREE.BoxGeometry(1, 1, 1);
      const material = new THREE.MeshStandardMaterial({
        color: texture ? 0xffffff : 0xff0000,
        map: texture,
        wireframe: isWireframe
      });
      
      const cube = new THREE.Mesh(geometry, material);
      scene.add(cube);
      objectRef.current = cube;
      
      // Pass the object reference to the parent component
      if (setObjectRef && setObjectRef.current !== undefined) {
        setObjectRef.current = cube;
      }
    };

    // Handle window resize
    const handleResize = () => {
      if (!canvasRef.current) return;
      
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
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

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      // Dispose textures, geometries and materials
      if (objectRef.current) {
        objectRef.current.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.geometry.dispose();
            if (Array.isArray(child.material)) {
              child.material.forEach(mat => {
                if (mat.map) mat.map.dispose();
                mat.dispose();
              });
            } else {
              if (child.material.map) child.material.map.dispose();
              child.material.dispose();
            }
          }
        });
      }
    };
  }, [obj, mtl, texture, isWireframe, setObjectRef, setLoadingStatus, setLoadingError]);

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

  return <canvas ref={canvasRef} className="w-full h-full" />;
};

export default ModelViewer;