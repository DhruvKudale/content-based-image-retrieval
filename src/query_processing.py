import cv2
import tqdm
from proximity import compare_basic_histograms, compare_split_histograms, compare_ccvs
from utility import load_images_from_folder, perform_eval

def process_query_image(query_image_path, images, distribution='basic-histogram',
                        proximity='l2', channel_bins=8):
    # Read the query image
    query_image = cv2.imread(query_image_path)

    # Compare the query image with each image from set of images
    results = []
    for image in tqdm.tqdm(images):
        candidate_image = cv2.imread(image)
        if distribution == 'split-histogram':
            corr, l1, l2 = compare_split_histograms(query_image, candidate_image, channel_bins)
        elif distribution == 'ccv':
            corr, l1, l2 = compare_ccvs(query_image, candidate_image, channel_bins, 300)
        else:
            corr, l1, l2 = compare_basic_histograms(query_image, candidate_image, channel_bins)
        entry = {}
        entry['image-name'] = image
        entry['l1'] = l1
        entry['l2'] = l2
        entry['corr'] = corr
        results.append(entry)

    # Sort the images based on proximity
    reverse = False
    if proximity == 'corr':
        reverse = True
    results.sort(key=lambda d: d[proximity], reverse=reverse)
    return results

def run_query(dataset_path, query_image_path, distribution, proximity, channel_bins, k,
              display_results=True, experimentation=False):
    # Load images and labels
    images, labels = load_images_from_folder(dataset_path)
    # Process the query image
    results = process_query_image(query_image_path, images, distribution, proximity, channel_bins)
    # Filter top K results
    if k < len(results):
        results = results[:k]

    # Experimentation to report p, r, f only if the flag is True
    p = 0
    r = 0
    f = 0
    if experimentation and len(labels):
        p, r, f = perform_eval(query_image_path, images, labels, results)
    #         print(f'Precision : {p:.4f}')
    #         print(f'Recall    : {r:.4f}')
    #         print(f'F1 Score  : {f:.4f}')
    return results, p, r, f

def run_query_inference_only(dataset_path, query_image_path, distribution, proximity, channel_bins, k,
              display_results=True, experimentation=False):
    # Load images and labels
    images, labels = load_images_from_folder(dataset_path)
    # Process the query image
    results = process_query_image(query_image_path, images, distribution, proximity, channel_bins)
    # Filter top K results
    if k < len(results):
        results = results[:k]

    # Return list of images and labels
    images = []
    labels = []
    for entry in results:
        images.append(entry['image-name'])
        labels.append(entry['image-name'].split('/')[-2:][0])
    return images, labels