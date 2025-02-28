# Drone-Detection-on-Cloud

เก็บรวบรวมโค้ดของ yolov8n พร้อมคำอธิบาย 

บทความที่วิเคราะห์และรีวิวสถาปัตยกรรมของ YOLOv8 - [A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS](https://arxiv.org/abs/2304.00501?utm_source=chatgpt.com)

Ultralytics YOLOv8 Docs - [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)

[![arXiv](https://img.shields.io/badge/arXiv-2304.00501-red)](https://arxiv.org/abs/2304.00501)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics%20YOLOv8-Docs-1E90FF)](https://docs.ultralytics.com/models/yolov8/)


# ขั้นตอนที่ 1: การติดตั้ง Dependencies

## 1.1. ติดตั้ง Python 3.8 - 3.10 - [Download python](https://www.python.org/downloads/)

## 1.2. ติดตั้ง Dependencies อื่นๆ
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install -U numpy opencv-python tqdm pandas matplotlib seaborn scipy
```

# ขั้นตอนที่ 2: การทำ Labels
โปรแกรมที่ใช้ทำ Labels - [labelImg](https://github.com/HumanSignal/labelImg)

[![Labels - labelImg](https://img.shields.io/badge/Labels%20-%20labelImg-FFD700)](https://github.com/HumanSignal/labelImg)

## 2.1. Install

```bash
pip install labelImg
pip install pyqt5 lxml
pip install pyqt5-tools #กรณีมีปัญหาเกี่ยวกับ PyQt ต้องติดตั้ง PyQt5-tools เพิ่ม
```
## 2.2. เรียกใช้งาน

```bash
cd <path โฟลเดอร์ labelImg-master>
python labelImg.py
```
จะขึ้นหน้านี้

![image](https://github.com/user-attachments/assets/6dde8d29-1572-4090-8572-e8348016ef5f)

## 2.3. การใช้งาน

### 2.3.1. กด Open Dir แล้วเลือกโฟลเดอร์ Dataset ที่ต้องการทำ Labels

![image](https://github.com/user-attachments/assets/515f5dc4-2f8e-47ec-acf2-4045bb20c3b0)

### 2.3.2. ตั้งค่าให้เป็น YOLO

![image](https://github.com/user-attachments/assets/089d0be5-5c01-4726-96d6-66ce4d12da46)

### 2.3.4. กด Creat ReactBox

![image](https://github.com/user-attachments/assets/469f4d62-970c-4b11-9d52-ce02f9a9a2cf)

### 2.3.5. ทำ Label เลือกคลาส แล้วกด Ok

![image](https://github.com/user-attachments/assets/ad30ed5a-a347-40f9-bd7b-c02aed0bb489)

### 2.3.6. กด Save 

![image](https://github.com/user-attachments/assets/311ef943-e7aa-4b3a-b1f5-76ee46d113f0)

### 2.3.7. กดไปรูปถัดไป

![image](https://github.com/user-attachments/assets/4897caf4-8b73-444c-acba-14ef219362b5)


### 2.3.8. ทำแบบนี้จนกว่าจะครบทุกภาพในโฟลเดอร์


# ขั้นตอนที่ 3: Training

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```
ตัวอย่างไฟล์ dataset.yaml - [dataset.yaml](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/dataset.yaml)

## 3.1. ในกรณีที่เริ่มเทรนใหม่
ตัวอย่าง: results = model.train(data=r"C:\Users\SETTHANUN\Desktop\Dear\Dataset\dataset.yaml", epochs=5, imgsz=640, project=r"C:\Users\SETTHANUN\Desktop\results", name="train")

```python
results = model.train(data=r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>", epochs=5, imgsz=640, project=r"<ใส่ path สำหรับเก็บไฟล์โมเดล>", name="<ใส่ชื่อโฟลเดอร์สำหรับเก็บไฟล์โมเดล>")
```

## 3.2. ในกรณีที่เทรนต่อจากโมเดลที่มีอยู่แล้ว
ตัวอย่าง: results = model.train(data=r"C:\Users\SETTHANUN\Desktop\Dear\Dataset\dataset.yaml", epochs=5, imgsz=640, project=r"C:\Users\SETTHANUN\Desktop\results", name="train", weights=r"C:\Users\SETTHANUN\runs\train\best.pt")

```python
results = model.train(data=r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>", epochs=5, imgsz=640, project=r"<ใส่ path สำหรับเก็บไฟล์โมเดล>", name="<ใส่ชื่อโฟลเดอร์สำหรับเก็บไฟล์โมเดล>", weights=r"<ใส่ path ที่เก็บไฟล์โมเดลที่เคยเทรนไว้แล้ว.pt>")
```

# ขั้นตอนที่ 4: Test
ตัวอย่าง: model = YOLO(r"C:\Users\SETTHANUN\runs\train\best.pt")
results = model.predict(source=r"C:\Users\SETTHANUN\Desktop\Dear\Dataset\DataSet_Pre_Cut_With_Label\Streak_All\Test", save=True, project=r"C:\Users\SETTHANUN\Desktop\results", name="test")
```python
model = YOLO(r"<ใส่ path ที่เก็บไฟล์โมเดลที่เทรนไว้แล้ว.pt>")
results = model.predict(source=r"<ใส่ path โฟลเดอร์ที่เก็บภาพสำหรับทดสอบ>", save=True, project=r"<ใส่ path ที่จะเก็บผลลัพธ์การทดสอบ>", name="<ใส่ชื่อโฟลเดอร์ที่จะเก็บผลลัพธ์การทดสอบ>")
```

# ขั้นตอนที่ 5: Result
## 5.1. การ Detect ธรรมดา
![image](https://github.com/user-attachments/assets/df236d60-8088-4cdf-b1c8-b86684b59f7b)

## 5.2. การ Detect และวาดจุด x,y,center
![image](https://github.com/user-attachments/assets/e4e3e3c9-f3d7-4504-8178-9d7e65284382)

## 5.2. ผลลัพธ์ในรูปแบบตาราง


![image](https://github.com/user-attachments/assets/ef1f9c51-e2ae-4296-9c96-77a767e8ba3e)


# ขั้นตอนที่ 6: เมื่อเกิดเหตุขัดข้อง
## 6.1. กรณีดาวน์โหลด Ultralytics ไม่ได้

โฟลเดอร์ใน Google drive - [Ultralytics drive](https://drive.google.com/file/d/1JaNYy7bcdA9FnZMclFmockTiUT2IHGE7/view?usp=sharing)

[![DRIVE - Ultralytics](https://img.shields.io/badge/DRIVE-Ultralytics-006400)](https://drive.google.com/file/d/1JaNYy7bcdA9FnZMclFmockTiUT2IHGE7/view?usp=sharing)

## 6.2. กรณีดาวน์โหลด labelImg ไม่ได้

โฟลเดอร์ใน Google drive - [labelImg drive](https://drive.google.com/file/d/1sQ2g4o0fdcOSwqGdM01ZhoKLkwvsYdpV/view?usp=sharing)

[![DRIVE - labelImg](https://img.shields.io/badge/DRIVE-labelImg-32CD32)](https://drive.google.com/file/d/1sQ2g4o0fdcOSwqGdM01ZhoKLkwvsYdpV/view?usp=sharing)

# เพิ่มเติม

ไฟล์ Jupyter notebook ที่เป็นโค้ดสำเร็จรูปแล้ว - [Fit yolo](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/Fit_yolo.ipynb)

ไฟล์โมเดลที่เทรนแล้ว - [Model](https://drive.google.com/file/d/1veqK1fydOkwu1toAbM0Fy2QSQefl48v0/view?usp=sharing)

ข้อมูลสำหรับการเทส - [Test](https://drive.google.com/file/d/1E1ZifJ56DEVDnBdfYzICRzc7HSHdTqY4/view?usp=sharing)

[![Fit yolo](https://img.shields.io/badge/Fit%20yolo-YOLOv8-90EE90)](https://github.com/Setthanun/YOLOv8_detection_streak/blob/main/Fit_yolo.ipynb) [![DRIVE - Model](https://img.shields.io/badge/DRIVE-Model-59ed17)](https://drive.google.com/file/d/1veqK1fydOkwu1toAbM0Fy2QSQefl48v0/view?usp=sharing)


