import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Camera Filters")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Filter state
        self.current_filter = "original"
        
        # Create video display
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # Create buttons
        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Original", command=lambda: self.set_filter("original")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Grayscale", command=lambda: self.set_filter("grayscale")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Edge Detection", command=lambda: self.set_filter("edge")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Segmentation", command=lambda: self.set_filter("segmentation")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Person Detection", command=lambda: self.set_filter("person")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Close", command=self.close).pack(side=tk.LEFT, padx=5)
        
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.update_video()
    
    def set_filter(self, filter_name):
        self.current_filter = filter_name
    
    def apply_grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def apply_edge_detection(self, frame):
        gray = self.apply_grayscale(frame)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def apply_segmentation(self, frame):
        pixel_values = frame.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_frame = centers[labels.flatten()]
        segmented_frame = segmented_frame.reshape(frame.shape)
        return segmented_frame
    
    def apply_person_detection(self, frame):
        frame_resized = cv2.resize(frame, (640, 480))
        boxes, weights = self.hog.detectMultiScale(frame_resized, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_resized, f'Persons: {len(boxes)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame_resized
    
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            if self.current_filter == "grayscale":
                frame = self.apply_grayscale(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for display
            elif self.current_filter == "edge":
                frame = self.apply_edge_detection(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self.current_filter == "segmentation":
                frame = self.apply_segmentation(frame)
            elif self.current_filter == "person":
                frame = self.apply_person_detection(frame)
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
            self.canvas.image = image  # Keep reference to avoid garbage collection

        self.window.after(10, self.update_video)
    
    def close(self):
        self.cap.release()
        self.window.destroy()

def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()