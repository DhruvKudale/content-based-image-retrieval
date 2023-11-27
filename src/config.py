# Set your input parameters

# Dataset folder path. All jpg images under it recursively are included in the dataset
dataset_path = "../dataset/test_set/"

# Query Image Folder path. All jpg images in this folder will be queried
query_image_path = "../dataset/just_query/"

# To select which frequency distribution is calculated it can be 'basic-histogram', 'split-histogram' or 'ccv'
# 'basic-histogram' is default value
distribution = 'basic-histo'

# To select which distance or similarity will be used to measure proximity of query image and canddiate image from dataset
# It can be 'l1', 'l2' or 'corr'. 'l2' is default
proximity = 'corr'

# This is to finalize how many bins are needed for histograms. If there are c bins per channel, the histogram will have c ** 3 bins
# Recommended to unchange this parameter. Default is 8
channel_bins = 16

# For indicating top 'k' images will be retrieved
k = 4

# If true will calculate P, R and F from the Korel dataset.
# For inference, it can be kept false (false by default)
experimentation = False

# To display results on matplotlib, this is to be kept True.
# This is True by default.
display_results = True