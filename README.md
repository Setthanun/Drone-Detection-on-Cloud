# Drone Detection on Cloud

## Step 1: Data Preparation

### 1.1. Download Dataset 
This dataset consists of high-resolution images capturing drones in various environments and conditions. The images are specifically collected for tasks such as object detection, tracking, and classification, enabling the development and evaluation of computer vision models for drone-related applications.

[![Dataset](https://img.shields.io/badge/Dataset-Drone-red)](https://drive.google.com/file/d/1EYZkrOq_FYzLHuo12X-vbcBQhEuoWPFc/view?usp=sharing)
### 1.2. Data Splitting
To provide flexibility in managing your dataset, you are encouraged to manually divide the data into three distinct sets: Training, Testing, and Validation. Please follow the steps below to properly organize your data
- Training Set: This set is used to train the model. It should contain the majority of the data and be representative of the overall dataset.
- Testing Set: The testing set is reserved for evaluating the model's performance after training. It should be kept separate from the training data to ensure a fair assessment of the model’s generalization ability.
- Validation Set: The validation set is used during the training process to tune hyperparameters and monitor the model’s performance. It helps in adjusting model parameters to avoid overfitting.

#### 1.2.1. Recommended Steps
- Randomization: Shuffle the data before splitting to ensure that the subsets are representative and free from any ordering bias.
- Manual Division: Allocate the appropriate number of samples to each of the three subsets, ensuring that each set is balanced and reflects the overall distribution of the data.

#### 1.2.2. Code for Data Splitting
The provided code is intended for data splitting tasks and is to be executed within the Command Prompt (CMD) interface.

- In order to run the code, please ensure that Python is called in the Command Prompt by executing the following command:
```bash
python
```

- Once Python is invoked, use the following code to perform the data splitting:
```bash
import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_size, val_size, test_size):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        return
    files = []
    for root, dirs, file_names in os.walk(source_dir):
        for file_name in file_names:
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                files.append(os.path.join(root, file_name))
    if not files:
        print(f"No image files found in the source directory.")
        return
    random.shuffle(files)
    total_files = len(files)
    train_count = int(total_files * train_size)
    val_count = int(total_files * val_size)
    test_count = total_files - train_count - val_count
    print(f'Total files: {total_files}, Train: {train_count}, Validation: {val_count}, Test: {test_count}')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for i, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        if i < train_count:
            target_dir = train_dir
        elif i < train_count + val_count:
            target_dir = val_dir
        else:
            target_dir = test_dir
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(file_path, target_path)
        print(f"Copied file {file_name} to {target_path}")

source_dir = 'path/to/your/dataset'
train_dir = 'path/to/save/train'
val_dir = 'path/to/save/val'
test_dir = 'path/to/save/test'

train_size =  #Specify the number you want to split for train
val_size =  #Specify the number you want to split for validation
test_size =  #Specify the number you want to split for test

split_dataset(source_dir, train_dir, val_dir, test_dir, train_size, val_size, test_size)
```

- Example of how to specify a path: 

```bash
source_dir = r'C:\Users\SETTHANUN\Downloads\archive\Database1\Dataset'
train_dir = r'C:\Users\SETTHANUN\Desktop\Split\Train\Image'
val_dir = r'C:\Users\SETTHANUN\Desktop\Split\Validation\Image'
test_dir = r'C:\Users\SETTHANUN\Desktop\Split\Test\Image'
```

### 1.3. Annotation
For the annotation process, we will use Roboflow, a powerful tool designed for labeling and annotating images efficiently. Roboflow allows us to create and manage labeled datasets for various machine learning tasks, including image classification, object detection, and segmentation. By leveraging its intuitive interface, we can easily annotate our images and export them in the required formats for model training and evaluation.

#### 1.3.1. Go to the Roboflow website.
[![Roboflow](https://img.shields.io/badge/roboflow-labels-purple)](https://app.roboflow.com/)

#### 1.3.2. On the Roboflow page, select Create my own workspace.
#### 1.3.3. Enter the desired workspace name in the 'Name Your Workspace' field.
#### 1.3.4. Choose Public Plan.
#### 1.3.5. Click Continue.

![image](https://github.com/user-attachments/assets/0b090fce-2aee-4874-a74c-7bc1990a5edc)

#### 1.3.6. On the Invite teammates page, you can add other users to collaborate on labeling.
#### 1.3.7. Click Create Workspace.

![image](https://github.com/user-attachments/assets/01cda051-d18a-4ff7-a3b0-6c144a10f087)

#### 1.3.8. Go to the Projects page and click New Project.
![image](https://github.com/user-attachments/assets/0fb03c02-4935-42be-aff3-083d5aae129a)

#### 1.3.9. Enter the Project Name and Annotation Group.
#### 1.3.10. Select Object Detection, then click Create Public Project.

![image](https://github.com/user-attachments/assets/5e4232b5-04f1-470b-8000-6d389755549c)

#### 1.3.11. On the Upload page, select 'Select Folder,' then choose the folder containing the images to be labeled.

![image](https://github.com/user-attachments/assets/b1bed0c5-7abc-4f49-984f-7e0bdc89aef9)

#### 1.3.12. Enter the Batch Name and then click Save and Continue

![image](https://github.com/user-attachments/assets/a9e840de-b077-454f-b167-59cf991cfdfb)



Models - [Ultralytics YOLO](https://docs.ultralytics.com/models/#featured-models)

[![Dataset](https://img.shields.io/badge/Dataset-Drone-red)](https://drive.google.com/file/d/1EYZkrOq_FYzLHuo12X-vbcBQhEuoWPFc/view?usp=sharing)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Models-YOLO-1E90FF)](https://docs.ultralytics.com/models/#featured-models)


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


