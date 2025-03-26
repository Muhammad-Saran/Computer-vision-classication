# Computer-vision-classication

This repository contains a **deep learning model** built with **TensorFlow** to classify images as either **dogs or cats**. The model is trained on the **Dog vs Cat dataset** and utilizes a simple neural network architecture for classification. A **predict script** is included to test the model on new images.

---

## **Installation Guide**

### **Step 1: Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies.

#### **Using `venv`**
```sh
mkdir dog_vs_cat_classifier
cd dog_vs_cat_classifier
python3 -m venv venv
```

#### **Activate the Virtual Environment**
- **Mac / Linux:**
  ```sh
  source venv/bin/activate
  ```
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```

---

### **Step 2: Install Dependencies**
First, install the required dependencies from `requirements.txt`:
```sh
pip install -r requirements.txt
```

If `requirements.txt` is missing, install the dependencies manually:
```sh
pip install tensorflow
pip install numpy
pip install pandas
pip install matplotlib
```

---

## **Dataset Download**
The model is trained using the **Dog vs Cat dataset** from Kaggle.  
🔗 **Download Dataset:** [Dog vs Cat Dataset](https://www.kaggle.com/datasets/arpitjain007/dog-vs-cat-fastai)  

### **Step 3: Extract and Specify Dataset Path**
1. **Download the dataset from Kaggle** and extract it.
2. **Specify the dataset path** in `train.py` before running the training script.

---

## **Usage**

### **Train the Model**
To train the model, run:
```sh
python train.py
```
This will:
- Load the dataset from the specified path.
- Train a convolutional neural network (CNN).
- Save the trained model as `model.h5`.

---

### **Test & Predict**
After training, use `predict.py` to test the model on a new image.

```sh
python predict.py --image path_to_your_image.jpg
```

Replace `path_to_your_image.jpg` with the actual path to the image you want to classify.

---

## **Customization**
You can modify:
- **`train.py`** to change the dataset path or tweak model parameters.
- **`predict.py`** to improve image preprocessing.
- **The dataset** to include additional images.

---

## **About**
This classifier is built using **TensorFlow** and a **Convolutional Neural Network (CNN)**.  
It is designed for **binary image classification** and can be fine-tuned or extended to other datasets.  

Feel free to contribute and enhance the project! 🚀

