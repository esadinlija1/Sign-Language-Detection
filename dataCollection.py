import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder="data/C"
counter=0

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("No Image from camera")
            continue

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = map(int, hand['bbox'])

            H, W = img.shape[:2]


            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(W, x + w + offset)
            y2 = min(H, y + h + offset)

            if x2 > x1 and y2 > y1:
                imgCrop = img[y1:y2, x1:x2]
                if imgCrop.size > 0:

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    ch, cw = imgCrop.shape[:2]
                    aspect = ch / cw

                    if aspect > 1:
                        k = imgSize / ch
                        new_w = int(np.ceil(cw * k))
                        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
                        imgResize = cv2.resize(imgCrop, (new_w, imgSize), interpolation=interp)
                        w_gap = (imgSize - new_w) // 2
                        imgWhite[:, w_gap:w_gap + new_w] = imgResize
                    else:
                        k = imgSize / cw
                        new_h = int(np.ceil(ch * k))
                        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
                        imgResize = cv2.resize(imgCrop, (imgSize, new_h), interpolation=interp)
                        h_gap = (imgSize - new_h) // 2
                        imgWhite[h_gap:h_gap + new_h, :] = imgResize

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)
                else:
                    print("Empty crop after slicing")
            else:
                print("Invalid ROI (after clamping)")

        cv2.imshow("Image", img)


        key = cv2.waitKey(1) & 0xFF

        if key==ord("s") and hands:
            counter+=1
            filename=f"{folder}/{counter}.png"
            cv2.imwrite(filename, imgWhite)
            print(counter)

        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
