# BrainTumorAI - Brain Tumor Detection using Artificial Intelligence

### How it Works

BrainTumorAI uses a convolutional neural network (CNN) architecture trained on a large dataset of labeled brain MRI scans. The model learns to recognize patterns indicative of brain tumors from the training data. During the inference phase,
the model analyzes the input image and identifies regions likely to contain tumors based on the learned patterns


## Dataset
The AI model is trained on a diverse and comprehensive dataset of brain MRI scans containing both tumor and non-tumor cases. 
The dataset is obtained from various medical institutions and is carefully curated to ensure accuracy and privacy compliance.

**Original Dataset - https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection**

## Usage

1: Clone the repository 
```https://github.com/Michael-RDev/BrainTumorAI.git```

2: Set up a Python virtual environment

```python -m venv BrainSolver``` 

***Activation***

> Windows : ```BrainSolver/Scripts/activate```

> Mac: ```source BrainSolver/bin/activate```



3: Install the required libraries by running the following command in the virtual environment:

```pip install -r requirements.txt```

4: Run the program

```python main.py```


## License
This project is licensed under the MIT License. 
Feel free to use, modify, and distribute the code for both commercial and non-commercial purposes.
