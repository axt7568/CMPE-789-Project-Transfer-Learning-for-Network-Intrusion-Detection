# CMPE-789-Project-Transfer-Learning-for-Network-Intrusion-Detection

**0. Documentation**
  - Please Refer to the [Presentation](https://docs.google.com/presentation/d/1BdeZTWzkFxyWEiXyWVWasaaS0RzX8snyaU6DZNnmQXI/edit?usp=sharing) for detailed explanation.

**1. Download Dataset**
  - Link to CSE-CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
  - Link to CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html

**2. Set Up Work Environment**
- Download and Install **PyCharm** using the following link: https://www.jetbrains.com/pycharm/download/#section=windows
- Setup Anaconda Virtual Environment using the link: https://docs.conda.io/projects/conda/en/latest/
  - Deep Learning Frameworks
	  - Download and Install **Tensorflow** using the link: https://www.tensorflow.org/install
	  - Download and Install **Keras** using the link: https://keras.io/
	  - Download and Install **Sklearn** using the link: https://scikit-learn.org/stable/install.html
	  - Download and Install **Seaborn** using the link: https://seaborn.pydata.org/installing.html
	  - Download and Install **Matplotlib** using the link: https://pypi.org/project/matplotlib/
	  - Download and Install **Numpy** using the link: https://numpy.org/install/
	  - Download and Install **Pandas** using the link: https://pandas.pydata.org/docs/getting_started/install.html
	  - Download and Install **Pickle** using the link: https://pypi.org/project/pickle5/
  - GPU Setup
    - GPU used: NVIDIA RTX 3070
    - Download and Install **CUDA** using the link: https://developer.nvidia.com/cuda-downloads
    - Download and Install **cuDNN** using the link: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

**3. Run and Evaluate Models**
  - Train and Evaluate Baseline Models
    ```
      - Run ./cnn_ids2017_baseline_combined.py and ./cnn_ids2018_baseline_combined.py
    ```
    
  - Evaluate No TF Baseline Models
    ```
      - Run ./ids2017c_no_tf_ids2018c.py and ./ids2018c_no_tf_ids2017c.py
    ```
    
  - Train and Evaluate Transfer Learning Models
    - For 2017 --> 2018 (Strategy 1 & 2) 
      ```
        - Run ./ids2017c_tf_ids2018c_s1_fz_fxl.py and ./ids2017c_tf_ids2018c_s2_fz_cxl.py
      ```
    - For 2018 --> 2017 (Strategy 1 & 2) 
      ```
        - Run ./ids2018c_tf_ids2017c_s1_fz_fxl.py and ./ids2018c_tf_ids2017c_s2_fz_cxl.py
      ```
    
**4. References**
  - https://www.kaggle.com/solarmainframe/ids-intrusion-csv
  - https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning
  - https://www.kaggle.com/code/azazurrehmanbutt/cicids-ids-2018-using-cnn
  - https://www.unb.ca/cic/datasets/ids-2018.html
  - https://machinelearningmastery.com/transfer-learning-for-deep-learning/
  - https://cs231n.github.io/transfer-learning/

