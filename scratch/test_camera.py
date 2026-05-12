import cv2

def test_camera():
    backends = [
        ("Default", None),
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF)
    ]
    
    for name, backend in backends:
        for index in range(2):
            print(f"Testing backend {name} with index {index}...")
            if backend is not None:
                cap = cv2.VideoCapture(index, backend)
            else:
                cap = cv2.VideoCapture(index)
                
            if cap.isOpened():
                print(f"  SUCCESS: Backend {name} with index {index} is working.")
                ret, frame = cap.read()
                if ret:
                    print(f"  Frame captured successfully.")
                else:
                    print(f"  Failed to read frame.")
                cap.release()
            else:
                print(f"  FAILED: Backend {name} with index {index} could not be opened.")

if __name__ == "__main__":
    test_camera()
