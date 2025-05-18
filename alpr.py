import cv2
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
img = cv2.imread('images/cargb.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 50))
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),5)
cv2.imshow("License Plate Detection", img)
cv2.waitKey()
img1 = cv2.imread('images/carinfrared.jpg')
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
for (x, y, w, h) in plates:
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0),5)
cv2.imshow("License Plate Detection", img1)
cv2.waitKey()
cv2.destroyAllWindows()