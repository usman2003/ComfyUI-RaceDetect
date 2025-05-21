
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace

class RaceDetectionNodeV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("race",)
    FUNCTION = "detect_race"
    CATEGORY = "Image Processing"

    def detect_race(self, image):
        """
        Detect race(s) in the input image using face_recognition and deepface.
        
        Args:
            image: ComfyUI IMAGE tensor (batch, height, width, channels).
        
        Returns:
            tuple: (str) Comma-separated list of detected races or error message.
        """
        # Convert ComfyUI tensor to NumPy array
        if len(image.shape) == 4:
            image = image.squeeze(0)  # Remove batch dimension if present
        image_np = image.numpy() if hasattr(image, 'numpy') else image
        image_np = (image_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        image_np = image_np[:, :, ::-1]  # Convert RGB to BGR for OpenCV

        # Detect faces using face_recognition
        boxes_face = face_recognition.face_locations(image_np)
        out_races = []

        if len(boxes_face) == 0:
            return ("No faces detected",)

        # Process each detected face
        for box_face in boxes_face:
            x0, y1, x1, y0 = box_face
            face_image = image_np[x0:x1, y0:y1]  # Extract face region (BGR)
            try:
                # Convert BGR to RGB for deepface
                face_image_rgb = face_image[:, :, ::-1]
                # Use deepface to predict race
                result = DeepFace.analyze(face_image_rgb, actions=['race'], enforce_detection=False)
                race = result[0]['dominant_race']
                out_races.append(race)
            except Exception as e:
                out_races.append(f"Error in race detection: {str(e)}")

        return (", ".join(out_races),)
