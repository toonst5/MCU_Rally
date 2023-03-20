import numpy as np
import cv2
import math
import time
#from adafruit_servokit import ServoKit


def PID(Kp, Ki, Kd, MV_bar=0):
    # initialize stored data
    e_prev = 0
    t_prev = -100
    I = 0
    # initial control
    MV = MV_bar
    
   
    while True:
        # yield MV, wait for new t, PV, SP
        e = yield MV
        print(e)
        P = Kp*e
        I = I + Ki*int(e)
        D = Kd*(e - e_prev)
        if -10>I:
            I=-10
        if I>10:
            I=10
        MV = MV_bar + P + I + D
     
        # update stored data for next iteration
        e_prev = e
        
def gstreamer_pipeline(
    capture_width=320,
    capture_height=240,
    display_width=320,
    display_height=240,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
R=0
L=0
Line1=0
Line2=0
v=0
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture('race.avi')
#'race.avi'
#kit = ServoKit(channels=16)
pid = PID(1,0.00000,1)
pid.send(None)

while(cap.isOpened()):
    program_starts = time.time()

    tic = time.perf_counter()
    ret, frame = cap.read()
    frame=cv2.resize(frame,(320,240))
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    mask = cv2.inRange(hsv, np.array([30,80,50]), np.array([255,255,180]))
    #maskEX = cv2.inRange(hsv, np.array([30,80,50]), np.array([255,255,255]))
    #mask=mask | maskEX
    #res = cv2.bitwise_and(frame, frame, mask = mask)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel, cv2.BORDER_REFLECT)
    
    b1 = False
    b2 = False
    b3 = False
    b4 = False
    cRX=0
    cR=0
    cL=0
    cLX=0
    cError=False
    for y in range(240):
        _Line = mask[239-y,300]
        if _Line>0:
            y2 = y
            b2 = True
            for y in range(40):
                if y2<=20:
                   cError=True
                   break;
                _Line = mask[239-y2-y+20,310]
                if _Line>0:
                    cR = 239-y2-y+20
                    cRX=310
                    break;
            break  
    
    if b2 == False:
        for y in range(240):
            _Line = mask[239-y,200]
            if _Line>0:
                y1 = y
                b1 = True
                for y in range(40):
                    if y1<=20:
                        cError=True
                        break;
                    _Line = mask[239-y1-y+20,210]
                    if _Line>0:
                        cR = 239-y1-y+20
                        cRX=210
                        break;
                break
                
    for y in range(240):
        _Line = mask[239-y,20]
        if _Line>0:
            y4 = y
            b4 = True
            for y in range(40):
                if y4<=20:
                    cError=True
                    break;
                _Line = mask[239-y4-y+20,10]
                if _Line>0:
                    cL = 239-y4-y+20
                    cLX=10
                    break;
            break
            
    if b4 == False:
        for y in range(240):
            _Line = mask[239-y,120]
            if _Line>0:
                y3 = y
                b3 = True
                for y in range(40):
                    if y<=20:
                        cError=True
                        break;
                    _Line = mask[239-y4-y+20,110]
                    if _Line>0:
                        cL = 239-y3-y+20
                        cLX=110
                        break;
                break
    if cL==0 or cR==0:
        cError=True
    print(cError)
    
    if b4 == True:
        L = y4
        Line1 = 20
    elif b3 == True:
        L = y3
        Line1 = 120
        
    if b2 == True:
        R = y2
        Line2 = 300
    elif b1 == True:
        R = y1
        Line2 = 200
    
    diff = R-L
    tang = abs(diff)/100
    
    if R>L:
        angle = (math.atan(tang) * 180 / math.pi)
    else:
        angle = 0 - (math.atan(tang) * 180 / math.pi)
        
    
    if (R>cR and L>cL) or (R<cR and L<cL):
        red=0
    else:
        red=255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    angle = round(angle,5)
    text = str(angle)
    
    
    mask = cv2.line(mask, (20,0), (20,240), (255,255,255), 2)
    mask = cv2.line(mask, (120,0), (120,240), (255,255,255), 2)
    mask = cv2.line(mask, (200,0), (200,240), (255,255,255), 2)
    mask = cv2.line(mask, (300,0), (300,240), (255,255,255), 2)
    
    
    frame = cv2.line(frame, (20,0), (20,240), (255,255,255), 2)
    frame = cv2.line(frame, (120,0), (120,240), (255,255,255), 2)
    frame = cv2.line(frame, (200,0), (200,240), (255,255,255), 2)
    frame = cv2.line(frame, (300,0), (300,240), (255,255,255), 2)
    
    frame = cv2.line(frame, (Line1,239-L), (Line2, 239-R), (255,0,red), 3)
    if cError==False:
        frame = cv2.line(frame, (Line1,239-L), (cLX, cL), (0,255,0), 3)
        frame = cv2.line(frame, (cRX, cR), (Line2, 239-R), (0,255,0), 3)
        mask = cv2.line(mask, (Line1,239-L), (cLX, cL), (0,255,0), 3)
        mask = cv2.line(mask, (cRX, cR), (Line2, 239-R), (0,255,0), 3)
    
    cv2.putText(mask, text, (20,210), font, 1, (255,255,255),2,cv2.LINE_4)
    
    
    test = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    test = cv2.line(test, (Line1,239-L), (Line2, 239-R), (255,0,0), 3)
    
    both = np.concatenate((frame,  test), axis=1)
    
    cv2.imshow('both',both)
    
    toc = time.perf_counter()
    
    #text = text.replace(".",",")
    #print((toc-tic))
    #print(text)
    
    #angle=(0-angle)+90
    print("in")
    print(v)
    v = angle
    MV = pid.send(v)*-1 +90   # compute manipulated variable

    
    print(MV)
    #t, PV, SP, TR = yield MV
    PV = angle
    #I = TR - MV_bar - P - D
    #P = Kp*(beta*SP - PV)
    #I = I + Ki*(SP - PV)*(t - t_prev)
    #eD = gamma*SP - PV
    #D = Kd*(eD - eD_prev)/(t - t_prev)
    #MV = MV_bar + P + I + D
    #eD_prev = eD
    #t_prev = t
 
    if MV > 180:
        MV = 179
    if 0 > MV:
        MV = 1 

    #kit.servo[8].angle = MV
    #kit.continuous_servo[9].throttle = 0.28#(1-(abs(MV-90)/90))/2
            
    print("It has been {0} seconds since the loop started".format(time.time() - program_starts))
    #cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
kit.servo[8].angle = 90
kit.continuous_servo[9].throttle = 0
