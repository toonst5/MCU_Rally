import numpy as np
import cv2
import math
import time
#from adafruit_servokit import ServoKit

show=True

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
    capture_width=80,
    capture_height=60,
    display_width=80,
    display_height=60,
    framerate=30,
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
gen=0
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture('race.avi')
#cap = cv2.VideoCapture(0)
#kit = ServoKit(channels=16)

pid = PID(1,0.00000,1)
pid.send(None)

width=80
widthD=int(width/2)
height=60
heightD=height-1
teller=0	
while(cap.isOpened()):
    program_starts = time.time()
    ret, frame = cap.read()
    if frame is None:
        cap = cv2.VideoCapture('race.avi')
        ret, frame = cap.read()

    frame=cv2.resize(frame, (width, height)) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)        
    lower_red = np.array([30,80,50])
    upper_red = np.array([255,255,180])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    Path = cv2.inRange(hsv, np.array([255,255,180]), upper_red)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel, cv2.BORDER_REFLECT)
    b1 = False
    b2 = False
    L=0
    R=0
    x2 = 0
    x=0
    y=heightD
    yM=height
    XO=widthD
    YO=heightD
    gem=0
    while (x < widthD):
    	while ((y >= 0)and((not b1) or (not b2))):
            _Line = mask[y,x]
            if (_Line>0 and not b1):
                #if (y==0):
                    #L=L-(heightD-y)
                #L=L+(heightD-y)
                Path[y,x]=250;
                b1=True
                if (yM>y):
                	yM=y
            _Line = mask[y,(x+widthD)]
            if (_Line>0 and not b2):
                #if (y==0):
                    #R=R-(heightD-y)
                #R=R+(heightD-y)
                Path[y,(x+widthD)]=250;
                b2=True
                if (yM>y):
                	yM=y
            y=y-1
    	b1=False
    	b2=False
    	x=x+1
    	y=heightD
    y=widthD
    yM=yM+4
    gen=0
    teller=0
    #Path=cv2.morphologyEx(Path,cv2.MORPH_CLOSE,(3,3))
    #Path=cv2.erode(Path,(3,3),iterations=1)
    while (y >= 0)and(y>yM):
    	x = gem
    	while (x>=0):
    		_Line = Path[y,x]
    		if(_Line>0):
    			break;
    		x=x-1
    	x2 = gem
    	while (x2<width):
    		_Line = Path[y,x2]
    		if(_Line>0):
    			break;
    		x2=x2+1
    	gem= int((((x2-widthD)-(widthD-x))/2)+widthD)
    	if gem>(XO+5):
    		gem=XO+5
    	elif gem<(XO-5):
    		gem=XO-5		
    	frame = cv2.line(frame, (XO,YO), (gem, y), (255,0,0), 1)
    	XO=gem
    	YO=y
    	y=y-1
    	x=0
    	x2=0
    	gen=gen+(gem-widthD)*y
    	teller=teller+0.25*y

    gen=(gen/(teller+1))+90
	
    angle = gen
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    angle = round(angle,5)
    text = str(angle)

    
    if show:
    	Path = cv2.line(Path, (widthD,0), (widthD,height), (255,255,255), 1)    
    	frame = cv2.line(frame, (widthD,0), (widthD,height), (255,255,255), 1)    
    	LineX=math.cos(math.radians(angle))*widthD
    	LineY=math.sin(math.radians(angle))*widthD
    	frame = cv2.line(frame, (widthD,height), ((widthD-int(LineX)), (widthD-int(LineY))), (255,0,0), 3)
    	cv2.putText(mask, text, (20,210), font, 1, (255,255,255),2,cv2.LINE_4)
    	test = cv2.cvtColor(Path, cv2.COLOR_GRAY2BGR)
    	test = cv2.line(test, (0,0), (0,height), (255,0,255), 1)
    	test = cv2.line(test, (width-1,0), (width-1,height), (255,0,255), 1)    
    	BW = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)    
    	test = cv2.line(test, (Line1,heightD-L), (Line2, heightD-R), (255,0,0), 1)    
    	both = np.concatenate((frame,  test), axis=1)
    	both = np.concatenate((both,  BW), axis=1)
    	showIMG=cv2.resize(both, (1008, 240)) 
    	cv2.imshow('showIMG',showIMG)
    
    #print(text)
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))
    
    #angle=(0-angle)+90
    v = angle
    
    MV = pid.send(v - 90)*-1 +90   # compute manipulated variable

    
    #print('pid')
    #print(-angle+MV)
    if MV > 180:
    	MV = 179
    if 0 > MV:
    	MV = 1 
    #kit.continuous_servo[9].throttle = 0.15

    #print(MV)
    #kit.servo[8].angle = MV
#     if MV>87 and MV<93:
#         kit.continuous_servo[9].throttle = 0.5
#     else:
#         kit.continuous_servo[9].throttle = 0.12
#            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
kit.servo[8].angle = 90
kit.continuous_servo[9].throttle = 0

