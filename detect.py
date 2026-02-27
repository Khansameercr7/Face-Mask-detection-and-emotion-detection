"""
detect.py  –  Real-Time Face Mask + Emotion Detection
======================================================
Uses OpenCV Haar cascade for face detection, then runs:
  • Mask model    → with_mask / without_mask
  • Emotion model → 7-class emotion

Controls:
  Q  → quit
  M  → toggle mask detection
  E  → toggle emotion detection
  S  → save screenshot

Run:
    python detect.py                    # webcam
    python detect.py --source video.mp4 # video file
    python detect.py --source image.jpg # single image
"""

import cv2, numpy as np, argparse, time
from pathlib import Path
import tensorflow as tf

MASK_MODEL_PATH    = 'models/mask_detector.h5'
EMOTION_MODEL_PATH = 'models/emotion_detector.h5'
CASCADE_PATH       = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MASK_SIZE    = (224, 224)
EMOTION_SIZE = (48,  48)

EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']
EMOTION_COLORS = {
    'angry':(50,50,220),'disgust':(50,150,50),'fear':(150,50,200),
    'happy':(50,220,50),'neutral':(200,200,200),'sad':(220,100,50),
    'surprise':(50,220,220),
}
MASK_COLOR   = (50, 220, 50)
NOMASK_COLOR = (50,  50, 220)


def load_models():
    m = {'mask': None, 'emotion': None}
    if Path(MASK_MODEL_PATH).exists():
        m['mask'] = tf.keras.models.load_model(MASK_MODEL_PATH)
        print(f"[OK] Mask model loaded")
    else:
        print(f"[!!] Mask model not found — run train_mask.py first")
    if Path(EMOTION_MODEL_PATH).exists():
        m['emotion'] = tf.keras.models.load_model(EMOTION_MODEL_PATH)
        print(f"[OK] Emotion model loaded")
    else:
        print(f"[!!] Emotion model not found — run train_emotion.py first")
    return m


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=10):
    x1,y1 = pt1; x2,y2 = pt2
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
    for cx,cy,ang in [(x1+r,y1+r,180),(x2-r,y1+r,270),(x1+r,y2-r,90),(x2-r,y2-r,0)]:
        cv2.ellipse(img,(cx,cy),(r,r),ang,0,90,color,thickness)


def draw_label(img, text, pos, color, conf=None):
    x,y = pos
    label = f"{text}  {conf:.0%}" if conf else text
    (w,h),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y0 = max(0, y-h-10)
    cv2.rectangle(img,(x,y0),(x+w+10,y),color,-1)
    cv2.putText(img,label,(x+5,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)


def draw_emotion_bars(img, probs, x, y):
    cv2.putText(img,'CONF',(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,150),1,cv2.LINE_AA)
    for i,(emo,p) in enumerate(zip(EMOTIONS,probs)):
        by = y+4+i*13
        cv2.rectangle(img,(x,by),(x+60,by+9),(40,40,40),-1)
        cv2.rectangle(img,(x,by),(x+int(p*60),by+9),EMOTION_COLORS[emo],-1)
        cv2.putText(img,emo[:3],(x+63,by+8),cv2.FONT_HERSHEY_SIMPLEX,0.3,(180,180,180),1,cv2.LINE_AA)


def run(source=0, show_mask=True, show_emotion=True):
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    mdl     = load_models()
    cap     = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open: {source}"); return

    t0,fc,fps = time.time(),0,0
    print("\nRunning... Q=quit M=mask E=emotion S=screenshot\n")

    while True:
        ok,frame = cap.read()
        if not ok: break
        H,W = frame.shape[:2]
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray,1.1,5,minSize=(60,60))

        for (fx,fy,fw,fh) in faces:
            roi = frame[fy:fy+fh, fx:fx+fw]
            if roi.size == 0: continue
            box_col = (100,255,100)

            if show_mask and mdl['mask']:
                inp  = cv2.resize(roi,MASK_SIZE)/255.0
                conf = float(mdl['mask'].predict(np.expand_dims(inp,0),verbose=0)[0][0])
                masked = conf > 0.5
                lbl    = 'MASK' if masked else 'NO MASK'
                c      = conf if masked else 1-conf
                col    = MASK_COLOR if masked else NOMASK_COLOR
                box_col = col
                draw_label(frame, lbl, (fx,fy), col, c)

            if show_emotion and mdl['emotion']:
                g    = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                inp  = cv2.resize(g,EMOTION_SIZE)/255.0
                inp  = np.expand_dims(np.expand_dims(inp,-1),0)
                prob = mdl['emotion'].predict(inp,verbose=0)[0]
                idx  = np.argmax(prob)
                ey   = fy-30 if show_mask and mdl['mask'] else fy
                draw_label(frame,EMOTIONS[idx].upper(),(fx,ey),EMOTION_COLORS[EMOTIONS[idx]],float(prob[idx]))
                draw_emotion_bars(frame,prob,fx+fw+5,fy+15)

            draw_rounded_rect(frame,(fx,fy),(fx+fw,fy+fh),box_col,2)
            for px,py in [(fx,fy),(fx+fw,fy),(fx,fy+fh),(fx+fw,fy+fh)]:
                dx = 1 if px==fx else -1; dy = 1 if py==fy else -1
                cv2.line(frame,(px,py),(px+dx*15,py),box_col,3)
                cv2.line(frame,(px,py),(px,py+dy*15),box_col,3)

        fc += 1
        if time.time()-t0 >= 1.0: fps=fc; fc=0; t0=time.time()
        cv2.rectangle(frame,(0,0),(W,30),(15,15,15),-1)
        cv2.putText(frame,"FACE ANALYSER",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,255),1,cv2.LINE_AA)
        cv2.putText(frame,f"FPS:{fps}",(W-80,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,100),1,cv2.LINE_AA)
        cv2.imshow('Face Analyser',frame)

        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        elif k==ord('m'): show_mask=not show_mask
        elif k==ord('e'): show_emotion=not show_emotion
        elif k==ord('s'):
            fn=f"reports/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fn,frame); print(f"Saved {fn}")

    cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',     default=0)
    ap.add_argument('--no-mask',    action='store_true')
    ap.add_argument('--no-emotion', action='store_true')
    args = ap.parse_args()
    src = args.source
    if str(src).isdigit(): src = int(src)
    run(src, not args.no_mask, not args.no_emotion)
