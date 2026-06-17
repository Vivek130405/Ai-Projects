import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import sys

# ==========================================
# 1. SETUP AND UTILITY FUNCTIONS
# ==========================================

def generate_dummy_rasenshuriken(filename="rasenshuriken.png"):
    """
    Generates a synthetic transparent PNG of a Rasenshuriken if one doesn't exist.
    This ensures the code is 100% runnable out of the box without external assets.
    """
    if os.path.exists(filename):
        return

    size = 400
    # Create an empty BGRA image (transparent background)
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)

    # Shuriken Blades - Anime blue/cyan energy (BGRA format)
    blade_color = (255, 230, 100, 220) 
    
    # 4 directional blades points
    p1 = np.array([[center[0], center[1]-40], [center[0]-160, center[1]-160], [center[0]-40, center[1]]])
    p2 = np.array([[center[0]+40, center[1]], [center[0]+160, center[1]-160], [center[0], center[1]-40]])
    p3 = np.array([[center[0], center[1]+40], [center[0]+160, center[1]+160], [center[0]+40, center[1]]])
    p4 = np.array([[center[0]-40, center[1]], [center[0]-160, center[1]+160], [center[0], center[1]+40]])
    
    # Draw blades
    cv2.fillPoly(img, [p1, p2, p3, p4], blade_color)

    # Core Energy Sphere
    # Outer ring
    cv2.circle(img, center, 80, (255, 200, 50, 180), 12)
    # Inner bright core (white/cyan)
    cv2.circle(img, center, 50, (255, 255, 220, 255), -1)

    # Energy swirls
    cv2.ellipse(img, center, (130, 30), 45, 0, 360, (255, 255, 255, 200), 4)
    cv2.ellipse(img, center, (130, 30), -45, 0, 360, (255, 255, 255, 200), 4)

    # Save to disk
    cv2.imwrite(filename, img)

def rotate_image(image, angle):
    """
    Rotates an image around its center using cv2.getRotationMatrix2D and cv2.warpAffine.
    Adjusts bounds so the image corners don't get cropped during rotation.
    """
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)

    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Calculate absolute cosine and sine to find new bounding dimensions
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Adjust rotation matrix to account for translation
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    # Perform the actual affine transformation
    rotated_img = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated_img

def is_hand_open(landmarks):
    """
    Determines if the hand is open or closed (fist) based on MediaPipe landmarks.
    Compares the distance of fingertips to the wrist vs MCP joints to the wrist.
    """
    wrist = landmarks.landmark[0]
    
    # Fingertips and corresponding MCP (base) joints
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    
    open_fingers = 0
    
    for tip, mcp in zip(tips, mcps):
        tip_node = landmarks.landmark[tip]
        mcp_node = landmarks.landmark[mcp]
        
        # Calculate Euclidean distances from wrist
        dist_tip = math.hypot(tip_node.x - wrist.x, tip_node.y - wrist.y)
        dist_mcp = math.hypot(mcp_node.x - wrist.x, mcp_node.y - wrist.y)
        
        # If the tip is further from the wrist than the MCP, the finger is extended
        if dist_tip > dist_mcp:
            open_fingers += 1
            
    # Consider the hand "open" if at least 3 fingers are extended
    return open_fingers >= 3

# ==========================================
# 2. MAIN APPLICATION CLASS
# ==========================================

class RasenshurikenAR:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Ensure asset exists and load it
        self.asset_path = "rasenshuriken.png"
        generate_dummy_rasenshuriken(self.asset_path)
        
        # Load as BGRA (4 channels)
        self.effect_img = cv2.imread(self.asset_path, cv2.IMREAD_UNCHANGED)
        if self.effect_img is None or self.effect_img.shape[2] != 4:
            print("Error: Could not load valid transparent PNG. Exiting.")
            sys.exit(1)

        # Animation variables
        self.rotation_angle = 0.0
        
        # Open Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera could not be opened. Check permissions or hardware.")
            sys.exit(1)

        # Optimize camera setting for FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def overlay_effect(self, frame, center_x, center_y, scale_size):
        """
        Overlays the Rasenshuriken effect with proper alpha blending, 
        Gaussian blur glow, and rotation.
        """
        frame_h, frame_w = frame.shape[:2]
        
        # 1. Resize the base image to the calculated scale
        resized_img = cv2.resize(self.effect_img, (scale_size, scale_size), interpolation=cv2.INTER_AREA)
        
        # 2. Rotate the image continuously
        rotated_img = rotate_image(resized_img, self.rotation_angle)
        
        # Update rotation for the next frame
        self.rotation_angle = (self.rotation_angle + 15) % 360  # Fast rotation for energy feel

        # Dimensions of the rotated image
        h, w = rotated_img.shape[:2]

        # Calculate placement bounds
        x1 = int(center_x - w / 2)
        y1 = int(center_y - h / 2)
        x2 = x1 + w
        y2 = y1 + h

        # 3. Handle screen boundary clipping
        bg_y1 = max(0, y1)
        bg_y2 = min(frame_h, y2)
        bg_x1 = max(0, x1)
        bg_x2 = min(frame_w, x2)

        ov_y1 = max(0, -y1)
        ov_y2 = h - max(0, y2 - frame_h)
        ov_x1 = max(0, -x1)
        ov_x2 = w - max(0, x2 - frame_w)

        # If completely off-screen, do nothing
        if bg_y1 >= bg_y2 or bg_x1 >= bg_x2:
            return frame

        # Extract Region of Interest (ROI) from frame and overlay
        bg_roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
        overlay_roi = rotated_img[ov_y1:ov_y2, ov_x1:ov_x2]

        # 4. Separate alpha and color channels
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha_3d = np.dstack((alpha, alpha, alpha))
        overlay_colors = overlay_roi[:, :, :3]

        # 5. Create Glow Effect (Anime Energy Style)
        # Multiply overlay colors by alpha to isolate the visible parts
        pre_multiplied_overlay = (alpha_3d * overlay_colors).astype(np.uint8)
        
        # Apply Gaussian blur to create the glowing aura
        glow = cv2.GaussianBlur(pre_multiplied_overlay, (45, 45), 0)
        
        # Add glow directly to the background ROI to illuminate the hand/background
        bg_with_glow = cv2.addWeighted(bg_roi, 1.0, glow, 1.8, 0.0)

        # 6. Proper Alpha Blending (RGBA)
        # Blend the sharp image on top of the glowing background
        blended = (alpha_3d * overlay_colors + (1.0 - alpha_3d) * bg_with_glow).astype(np.uint8)

        # Apply back to the main frame
        frame[bg_y1:bg_y2, bg_x1:bg_x2] = blended
        return frame

    def run(self):
        """
        Main application loop.
        """
        prev_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to grab frame. Exiting.")
                    break

                # Flip frame horizontally for a mirror effect (more natural AR)
                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]

                # Convert to RGB for MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                # Process hand detections
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        
                        # Check if hand is open or a fist
                        if is_hand_open(hand_landmarks):
                            # Calculate Palm Center
                            # Using midpoint between wrist (0) and middle finger MCP (9)
                            wrist = hand_landmarks.landmark[0]
                            mid_mcp = hand_landmarks.landmark[9]
                            
                            center_x = int((wrist.x + mid_mcp.x) / 2 * frame_w)
                            center_y = int((wrist.y + mid_mcp.y) / 2 * frame_h)

                            # Calculate hand scale/size dynamically based on wrist-to-mcp distance
                            hand_dist = math.hypot(wrist.x - mid_mcp.x, wrist.y - mid_mcp.y)
                            base_size = int(hand_dist * frame_w * 4.5) # Scaling multiplier
                            
                            # Constrain size to prevent crash on extreme close-ups
                            base_size = max(50, min(base_size, 800))

                            # Add slight pulsing scale animation based on time
                            current_time = time.time()
                            pulse = 1.0 + 0.1 * math.sin(current_time * 8.0)
                            final_size = int(base_size * pulse)

                            # Apply the AR effect
                            frame = self.overlay_effect(frame, center_x, center_y, final_size)
                        
                        # Optional: Draw hand landmarks for debugging/tech-look
                        # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Calculate and display FPS to ensure smooth performance
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Open hand: Rasenshuriken | Fist: Hide | 'q': Quit", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display the output
                cv2.imshow("Naruto Rasenshuriken AR", frame)

                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        finally:
            # Clean up and release resources
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    app = RasenshurikenAR()
    app.run()