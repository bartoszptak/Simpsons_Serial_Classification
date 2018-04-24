import cv2

file_path = "data/simpsons.mp4"
window_name = "Simpshons Classification"



cap = cv2.VideoCapture(file_path)


while cap.isOpened():
    ret,frame = cap.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Test  (69.69%)", (30, 30), font, 1, (0, 0, 150), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()