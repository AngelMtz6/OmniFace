import cv2

for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]:
    cap = cv2.VideoCapture(0, backend)
    ok, frame = cap.read()
    print(f"{name}: {'OK ' + str(frame.shape) if ok else 'FALLO'}")
    cap.release()
