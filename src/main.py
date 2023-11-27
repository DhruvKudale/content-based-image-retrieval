import tqdm
import glob
import cv2
from matplotlib import pyplot as plt
# Local imports
from query_processing import run_query, run_query_inference_only
from config import (dataset_path, query_image_path, distribution, proximity,
                    channel_bins, k, display_results, experimentation)
def display_images(query_image, matched_image_path, proximity_value):
    # Display query image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    # Display matched image
    plt.subplot(1, 2, 2)
    matched_image = cv2.imread(matched_image_path)
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Matched with value: {proximity_value:.3f}")
    plt.show()

def diplay_results_on_grid(results, query_image_path):
    n = len(results)
    print(f"Query image: {query_image_path}")
    print("Top Matched Images from the dataset:")
    query_image = cv2.imread(query_image_path)
    plt.subplot(1, n + 1, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    for i in range(len(results)):
        entry = results[i]
        print(f"{entry['image-name']}")
        print(f"{distribution} using {proximity} : {entry[proximity]:.3f}\n")
        # Display matched image
        matched_image_path = entry['image-name']
        proximity_value = entry[proximity]
        plt.subplot(1, n + 1, i + 2)
        matched_image = cv2.imread(matched_image_path)
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Matched with value: {proximity_value:.3f}")
    plt.show()

def perfrom_cbir_experiment(dataset_path, query_images_path, distribution,
                 proximity, channel_bins, k, experimentation, display_results):
    query_images = glob.glob(query_images_path + '/*.jpg')
    total_p = 0
    total_r = 0
    total_f = 0
    total_q = len(query_images)
    print(total_q)
    
    for query_image_path in tqdm.tqdm(query_images):
        # Run the query and get results and p, r, f for top k predicitons
        results, p, r, f = run_query(dataset_path, query_image_path, distribution, proximity,
                        channel_bins, k, display_results = display_results, experimentation = experimentation)
        # Update values for experimentation of a set of query images
        total_p = total_p + p
        total_r = total_r + r
        total_f = total_f + f
        # Display the results
        if display_results:
            diplay_results_on_grid(results, query_image_path)
    if experimentation:
        print(f'Precision : {total_p/total_q:.3f}')
        print(f'Recall    : {total_r/total_q:.3f}')
        print(f'F1 Score  : {total_f/total_q:.3f}')


def perfrom_cbir(dataset_path, query_image_path, distribution,
                            proximity, channel_bins, k, experimentation, display_results):
    # Run the query and get results

    images, labels = run_query_inference_only(dataset_path, query_image_path, distribution, proximity,
                                 channel_bins, k, display_results=display_results, experimentation = False)
    return images, labels

if __name__ == "__main__":
    # Calling the core function. Arguments are picked from config file.
    # See the config file for more details on the arguments
    # images, labels = perfrom_cbir(dataset_path = dataset_path, query_image_path = query_image_path,
    #              distribution = distribution, proximity = proximity, channel_bins = channel_bins,
    #              k = k, experimentation = experimentation, display_results = display_results)
    #
    # print(images)
    # print(labels)

    import time

    t0 = time.time()
    perfrom_cbir_experiment(dataset_path=dataset_path, query_images_path=query_image_path,
                            distribution=distribution, proximity=proximity, channel_bins=channel_bins,
                            k=k, experimentation=experimentation, display_results=False)
    t1 = time.time()

    total = t1 - t0
    print(total)

