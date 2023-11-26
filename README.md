# CBIR
Content Based Image Retrieval (CS 663 Project)

### Instructions to run
```
python src/main.py
```
This calls the perfrom_cbir() which has the following input parameters.
#### Set your input parameters in src/config.py

#### 1) Dataset folder path. 
All jpg images under it recursively are included in the dataset

#### 2) Query Image Folder path. 
All jpg images in this folder will be queried to the above dataset

#### 3) Distribution to Determine
Select which frequency distribution is calculated it can be 'basic-histogram', 'split-histogram' or 'ccv'.
'basic-histogram' is default value

#### 4) Proximity to Measure
To select which distance or similarity will be used to measure proximity of query image and canddiate image from dataset
It can be 'l1', 'l2' or 'corr'. 'l2' is default

#### 5) Channel Bins
This is to finalize how many bins are needed for histograms. If there are c bins per channel, the histogram will have c ** 3 bins
Recommended to unchange this parameter. Default is 16

#### 6) k
For indicating top 'k' images will be retrieved

#### 7) Experimentation Flag
If true will calculate P, R and F from the Korel dataset.
For inference, it can be kept false. This flag is false by default)

#### 8) Display_results Flag 
To display results on matplotlib, this is to be kept True.
This is True by default.
