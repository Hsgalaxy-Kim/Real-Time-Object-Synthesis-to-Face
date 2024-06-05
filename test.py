import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

mask1 = cv2.imread('img\mask_crop.jpg')
gray_mask = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
mask1[gray_mask>240] = 0
MH, MW, _ = mask1.shape
mask2 = mask1.copy()
# cv2.imshow("camera", mask1)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

skin_set = 0
lib_set = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if cap.isOpened():# 카메라 실행 여부
    ret, a = cap.read() # a: 영상 프레임
    while ret:
        ret, a = cap.read()
        if skin_set == 0:
            x,y,w,h = cv2.selectROI('img', a, False)
            skin1 = a[y:y+h, x:x+w]
            hsv_skin1 = cv2.cvtColor(skin1, cv2.COLOR_BGR2HSV)
            hist_roi_skin = cv2.calcHist([hsv_skin1],[0, 1], None, [180, 256], [0, 180, 0, 255] )
            skin_set = 1
        if lib_set == 0:
            x,y,w,h = cv2.selectROI('img', a, False)
            lib1 = a[y:y+h, x:x+w]
            LH, LW, _ = lib1.shape
            lib2 = lib1.copy()
            lib_set = 1
        
        # start = time.time()
        
        hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        bp_hsv = cv2.calcBackProject([hsv], [0, 1], hist_roi_skin,  [0, 180, 0, 255], 1)
        bp_hsv[bp_hsv>0] = 255
        # print("backproject:", time.time()- start)

        bp_hsv = cv2.dilate(bp_hsv, np.ones((20,20),np.uint8), iterations=1) #모폴로지 연산으로 잡음제거, 너무 타이트하면(10,10) 오히려 불안정해짐.
        bp_hsv = cv2.erode(bp_hsv, np.ones((40,10),np.uint8), iterations=1) #모폴로지 연산으로 잡음제거
        bp_hsv = cv2.dilate(bp_hsv, np.ones((40,10),np.uint8), iterations=1) #모폴로지 연산으로 잡음제거
        # print("morphology:", time.time()- start)
        
        mask = cv2.copyTo(a, bp_hsv) #이미지에 투영
        
        contours, _ = cv2.findContours(bp_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(mask, contours, -1, (255,0,0), 4) #모든 contour 그리기
        
        for contour in contours:
            area = cv2.contourArea(contour) #contour 크기 계산
            if area > 10000: # 일정 크기 이상만 허용
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(mask, (x,y), (x+w, y+h), (0,255,0), 2)
                # print("contour:", time.time()- start)

                th, tw, _ = lib2.shape
                res = cv2.matchTemplate(mask[y:int((y+h)*4/5), x:x+w], lib2, cv2.TM_CCOEFF_NORMED) # 템플릿 매칭
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # 히트맵에서의 최솟값, 최댓값 구하기
                top_left = max_loc
                match_val = max_val
                bottom_right = (top_left[0] + x + tw, top_left[1] + y + th)
                center_x, center_y =  x + top_left[0] + int(LW/2), y + top_left[1] + int(LH/2) #입의 중심 좌표
                cv2.rectangle(a, (top_left[0] + x, top_left[1]+ y), bottom_right, (0,0,255),2)
                # print("matchTemplate:", time.time()- start)
                
                #마스크 씌우기
                mask2 = mask1.copy()
                mask2 =cv2.resize(mask2, (int(w/1.3/2)*2, int(w/MW*MH/1.3/2)*2)) #w ,h
                mh, mw, _ = mask2.shape
                a[center_y-int(mh/2) : center_y+int(mh/2), center_x - int(mw/2):center_x + int(mw/2)][mask2>10] = mask2[mask2>10]
                # print("mask:", time.time()- start)
                break
        cv2.imshow("camera", a)
        if cv2.waitKey(1) & 0xFF == 27:
            break# 종료 커맨드.
cap.release()
cv2.destroyAllWindows()