import cv2


cap = cv2.VideoCapture(0)
cv2.namedWindow('test')


for i in range(5000):

    ret, frame =cap.read()

    smaller_frame = frame[100:400,50:300]
    smaller_frame =cv2.cvtColor(smaller_frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('test',smaller_frame)
    print(f"frame number: {i}")
    cv2.imwrite(f'C:/Users/KEREM/Desktop/5/image_{i}.png',smaller_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()