import json, time, cv2, torch, numpy as np
from collections import deque
from torchvision import transforms
from cvzone.HandTrackingModule import HandDetector
import timm

ART_DIR = "artifacts"
MODEL_PATH = f"{ART_DIR}/model_best.pth"
CLASSES_PATH = f"{ART_DIR}/classes.json"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.60
SMOOTH_N = 5


with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)
num_classes = len(CLASSES)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

detector = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.5)
cap = cv2.VideoCapture(0)

pred_q = deque(maxlen=SMOOTH_N)
fps_t = time.time(); fps = 0.0

def letterbox_crop(img, bbox, size=IMG_SIZE, offset=20):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x1 = max(0, x - offset); y1 = max(0, y - offset)
    x2 = min(W, x + w + offset); y2 = min(H, y + h + offset)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]  # BGR
    ch, cw = crop.shape[:2]
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if ch >= cw:
        k = size / ch
        new_w = max(1, int(round(cw * k)))
        crop_r = cv2.resize(crop, (new_w, size), interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
        x_start = (size - new_w)//2
        canvas[:, x_start:x_start+new_w] = crop_r
    else:
        k = size / cw
        new_h = max(1, int(round(ch * k)))
        crop_r = cv2.resize(crop, (size, new_h), interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
        y_start = (size - new_h)//2
        canvas[y_start:y_start+new_h, :] = crop_r
    return canvas

try:
    while True:
        ok, frame = cap.read()
        if not ok: break

        hands, img = detector.findHands(frame)
        label_txt = "—"
        pmax = 0.0

        if hands:
            hand = hands[0]
            crop = letterbox_crop(frame, hand['bbox'], size=IMG_SIZE, offset=20)
            if crop is not None:
                # BGR->RGB, transform, infer
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                x = tfm(rgb).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                i = int(np.argmax(probs)); pmax = float(probs[i])
                pred_q.append(i)


                if len(pred_q) == pred_q.maxlen:
                    counts = np.bincount(np.array(pred_q), minlength=num_classes)
                    i = int(counts.argmax())

                label_txt = CLASSES[i] if pmax >= THRESH else "unknown"

                cv2.imshow("HandCrop", crop)

        now = time.time()
        if now - fps_t >= 0.5:
            fps = 2.0 / (now - fps_t)
            fps_t = now

        cv2.putText(img, f"{label_txt} ({pmax:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Realtime Sign Classifier", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()