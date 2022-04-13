# NetCC_Server
[2020 NetCC] Smart Navigation using School zone CCTVs: Server &amp; Client code    
_Update: 2022-04-13_  
## **Index**
+ [About this project](##-**about-this-project**)
+ [Overview](#-**overview**)
  + [Goal](#-**goal**)
  + [Flow](#-**flow**)
+ [Detail Function](#-**detail-function**)
  + [Server](#-**server**)
  + [Client](#-**client**)
+ [Environment](#-**environment**)

## **About this project**
<이미지 삽입>
+ 프로젝트 이름: 스쿨존 CCTV를 활용한 스마트 내비게이션  
+ 프로젝트 진행 목적: 공모전 출전   
**Net Challenge 시즌 7**   
(참고 url: https://www.koren.kr/kor/Alram/contyView.asp?s=1&page=1)    


+ 프로젝트 진행 기간: 2020년 5월 ~ 2020년 12월


+ 프로젝트 참여 인원: 5명  
## **Overview** 
> ### **Goal**
+ (목적) 스쿨존 내 CCTV를 활용해 실시간으로 어린이 보행자 위치를 파악함으써 사고를 방지하고 인명피해를 줄이기 위함. 
+ (필요성) 운전자에게 스쿨존 내 보행자 위치를 내비게이션 지도상에 나타냄으로써 사고를 선제 예방할 수 있음. 
> ### **Flow**
<이미지 삽입>  

## **Detail Function**
> ### **Server**   
> #### **여러 Client(CCTV)로부터 전송된 위치 좌표의 값을 최종적으로 저장할 배열을 생성함. (Merging)**
> #### Server.py


``` python
temp[num]=loc_tmp

data=conn.recv(1024); 
point=data.decode() #클라이언트에서 수신받은 GPS 배열

loction=point.split() # point 값을 loction에 저장 (비교용) 
loc_tmp=point.split() # point 값을 loc_tmp에 저장 후 temp[num]에 저장(line 36) 

#temp와 location 비교
for i in range(int(tem_len)):
    for j in range(int(loc_len)):
        
        tem_lat = float(temp[cnt][2*i]); tem_long = float(temp[cnt][2*i+1])
        loc_lat = float(location[2*j]); loc_long = float(location[2*j+1])
        
        dist = math.sqrt((tem_lat-loc_lat)**2+(tem_long-loc_long)**2)
       
        # 입력된 좌표의 거리를 비교하여 특정 거리 (0.00003) 내에 위치할때 한 사람의 좌표로 인식 -> merge
        if (dist < 0.00003) and (dist < min):
            min = dist
            lat = (tem_lat+loc_lat)/2; long = (tem_long+loc_long)/2
            location[2*j] = str(lat); location[2*j+1] = str(long)
      
        else: state = 1
   
    if state == 1: #최종 배열 생성
        location.append(temp[cnt][2*i]); location.append(temp[cnt][2*i+1])
        state = 0
  temp[cnt]=0
```
+ 이미지 있으면 삽입
+ 여러 Client(CCTV)에서 수신받은 데이터를 각각 다른 변수에 저장함. 
+ 서로 다른 변수에 저장된 위치좌표 데이더들을 비교하여 가장 가까운 거리에 있는 좌표끼리의(한 사람으로 간주) 평균을 내어 최종 배열에 담아줌. 

> ### **Client**   
> #### **이미지에서 인식된 사람의 픽셀 좌표를 GPS 데이터로(위도, 경도) 변환함.**  
> #### **필요한 데이터: CCTV의 위도 경도 값**  
> #### CCTV 1.py, CCTV 2.py  


#### **(1) Detect**   
``` python
#Detect Function
detected, _ = hog.detectMultiScale(frame) # 사람 detect
  
for (x, y, w, h) in detected:
    cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
    foot_x = x+(w/2); foot_y = y+(h-60) # 추출한 사람의 발끝 좌표 
    perspective(frame, foot_x, foot_y) # 변환 함수 호출
```
입력된 이미지에서 사람을 인식하고, 인식된 사람에 사각형 프레임을 씌우면서 사람의 발 끝 좌표를 추출함.   
추출된 좌표에 대해서 변환 함수를 호출함.  


#### **(2) Perspective**  
``` python
M = cv2.getPerspectiveTransform(pts1,pts2) #기울어진 화면을 평평하게 perspective transform 

x= np.mat([[M[0][0],M[0][1],M[0][2]]])
y= np.mat([[M[1][0],M[1][1],M[1][2]]])
b= np.mat([[M[2][0],M[2][1],M[2][2]]])

c= np.mat([[f_x],[f_y],[1]])

#사람의 픽셀 좌표 추출
x_map = int((x*c)/(b*c))
y_map = int((y*c)/(b*c))

point = gps_conversion(x_map, y_map) # 사람의 픽셀 좌표를 gps 장 위치 좌표로 변환   
```
기울어진 이미지를 Perspective Transform 한 후, 변환 행렬 M 생성함.  
생성된 배열에 곱 연산을 통해 기울어진 사람의 픽셀좌표를 평평한 좌표 값으로 변환하여 GPS 좌표 변환함수에 넣음.    


#### **(3) GPS_Conversion**  
``` python
a = Symbol('a')
b = Symbol('b')

equation1 = (gps_list[0][1]-a)**2+(gps_list[0][0]-b)**2-(foot_x*x_rate)**2
equation2 = ((gps_list[0][0]-gps_list[1][0])/(gps_list[0][1]-gps_list[1][1]))*(a-gps_list[0][1])+gps_list[0][0]-b
res=solve((equation1, equation2), dict=True)
    
x = Symbol('x')
y = Symbol('y')

equation1 = (res[1][a]-x)**2+(res[1][b]-y)**2-(foot_y*y_rate)**2
equation2 = (-(gps_list[0][1]-gps_list[1][1])/(gps_list[0][0]-gps_list[1][0]))*(x-res[1][a])+res[1][b]-y
res=solve((equation1, equation2), dict=True)
gps = str(res[0][y])+' '+str(res[0][x])+' '
```
(이미지 삽입)   
(식 추가)  
직선의 방정식을 세워 각 점의 좌표를 구하는 방식으로 문제를 풀었음.  

## **Environment** 
+ python 3.7.3
