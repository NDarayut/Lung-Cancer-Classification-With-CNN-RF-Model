Dataset can be found on [kaggle](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)   
<h2>How to setup</h2>  
Install virtual environment:<code>pip install virtualenv</code>  

Create virtual environment:<code>python<version> -m venv <virtual-environment-name></code>  
  
Activate virtual environment:<code>.\virtual-environment-name\Scripts\Activate</code>  

Install packages:<code>pip install -r requirements.txt </code>  

Run program:<code>py app.py</code>  

Get address  

Deactivate environment:<code>deactivate</code>  


<h2>Abstract </h2>   
Lung cancer is a type of cancer that starts when abnormal cells grow in an uncontrolled way in the lungs. It is a serious health issue that can cause severe harm and death. Cancer that is caught at an early stage can be treated and could potentially saves lives. However, only 16% of lung cancer are diagnosed at an early stage, meaning more than 80% of lung cancer are diagnosed at a much later stage which could drastically reduce the survival rate of lung cancer patient. In this research, deep-learning and machine learning method is used to accurately identify the type of nodules within the lungs by using CT-images as input. CT-scan is one of the methods used to identify lung cancer, but radiologist struggle to identify the cancerous tumor residing in the lungs. With the help of technology and Artificial Intelligence, radiologist can use these tools to assist them in identifying the type of tumor and could further decreased the mortality rate of lung cancer. Through this research a dataset collected from the Iraqi hospitals was used on the hybrid convolutional neural network and random forest model (CNN-RF) to classify the type of nodule: benign, normal or malignant. The proposed model gives high accuracy ups to 94% on the testing set. The other performance metrices comes with high values such as 93% on recall average and 95% on precision average.   
<h2>About Dataset</h2>   
The Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD) lung cancer dataset was collected in the above-mentioned specialist hospitals over a period of three months in fall 2019. It includes CT scans of patients diagnosed with lung cancer in different stages, as well as healthy subjects. IQ-OTH/NCCD slides were marked by oncologists and radiologists in these two centers. The dataset contains a total of 1190 images representing CT scan slices of 110 cases. These cases are grouped into three classes: normal, benign, and malignant. of these, 40 cases are diagnosed as malignant; 15 cases diagnosed with benign; and 55 cases classified as normal cases. The CT scans were originally collected in DICOM format. The scanner used is SOMATOM from Siemens. CT protocol includes: 120 kV, slice thickness of 1 mm, with window width ranging from 350 to 1200 HU and window center from 50 to 600 were used for reading. with breath hold at full inspiration. All images were de-identified before performing analysis. Written consent was waived by the oversight review board. The study was approved by the institutional review board of participating medical centers. Each scan contains several slices. The number of these slices range from 80 to 200 slices, each of them represents an image of the human chest with different sides and angles. The 110 cases vary in gender, age, educational attainment, area of residence and living status. Some of them are employees of the Iraqi ministries of Transport and Oil, others are farmers and gainers. Most of them come from places in the middle region of Iraq, particularly, the provinces of Baghdad, Wasit, Diyala, Salahuddin, and Babylon.   
<h2>References</h2>   
(alyasriy, hamdalla; AL-Huseiny, Muayed (2023), “The IQ-OTH/NCCD lung cancer dataset”, Mendeley Data, V4, doi: 10.17632/bhmdr45bh2.4)   
