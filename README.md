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

#### 1.3.13. Click Start Manual Labeling. 
![image](https://github.com/user-attachments/assets/e8752296-f373-4d92-91c6-53bcb1bbae30)

#### 1.3.14. You can add other users and then click 'Assign to Myself'.

![image](https://github.com/user-attachments/assets/faa46488-2913-4db5-ae02-6c2952b62660)

#### 1.3.15. Select Start Annotating

![image](https://github.com/user-attachments/assets/db8241e6-a443-459f-bbd3-ffa435e46ccf)

#### 1.3.16. Select the Bounding Box Tool, then perform the annotation by dragging a rectangle to surround the drone.
#### 1.3.17. There will be a field to enter the class name, type 'Drone', and then click Save. Repeat this process until all images are done. **The class name must be the same for all images.

![image](https://github.com/user-attachments/assets/cb10a487-069f-4fe6-a457-5cf21c5e65ea)

#### !!! Don't forget to specify whether it's training, validation, or test data.
![image](https://github.com/user-attachments/assets/2513afa2-becd-4500-8789-f8d0364026e9)

#### 1.3.18. After annotating, click + New Version.

![image](https://github.com/user-attachments/assets/7d552e4e-8292-43d0-b073-90397edb01a2)

#### 1.3.19. Check the accuracy of the data, such as the total number of images, the number of classes, and the number of images in each split.

![image](https://github.com/user-attachments/assets/2bb176b6-a349-4f28-ba33-dbaf3bb2726e)

#### 1.3.20. Preprocessing: Remove Auto-Orient and Resize, then click Continue.
![image](https://github.com/user-attachments/assets/eb651c41-b553-4035-8186-d009cbed2092)

#### 1.3.21. Augmentation: Don't select anything, click Continue.

![image](https://github.com/user-attachments/assets/31fb6faa-9a03-4d85-a300-b5314c76dc3d)

#### 1.3.22. Create: Click Create.

![image](https://github.com/user-attachments/assets/b118f76f-07d9-44b9-a0d9-ed5b7a1f0d97)

#### 1.3.23. On the Dataset Versions page, select Download Dataset.

![image](https://github.com/user-attachments/assets/34cbcb65-173b-49b0-a1d3-2270d1b4e1e4)

#### 1.3.24. On the Download page, select 'Download zip to computer', choose the format as TXT, then select the model name and click Continue.

![image](https://github.com/user-attachments/assets/c4c48a03-44e5-40fd-b3a4-7c5023c0d13a)

