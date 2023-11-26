import tqdm
import glob

# Local imports
from query_processing import run_query
from config import (dataset_path, query_image_path, distribution, proximity,
                    channel_bins, k, display_results, experimentation)

def perfrom_cbir(dataset_path, query_images_path, distribution,
                 proximity, channel_bins, k, experimentation, display_results):
    query_images = glob.glob(query_images_path + '*.jpg')
    # For corr
    query_images = query_images[:1]
    total_p = 0
    total_r = 0
    total_f = 0
    total_q = len(query_images)
    
    for query_image_path in tqdm.tqdm(query_images):
        p, r, f = run_query(dataset_path, query_image_path, distribution, proximity, 
                        channel_bins, k, display_results = display_results, experimentation = experimentation)
        total_p = total_p + p
        total_r = total_r + r
        total_f = total_f + f
    
    if experimentation:
        print(f'Precision : {total_p/total_q:.3f}')
        print(f'Recall    : {total_r/total_q:.3f}')
        print(f'F1 Score  : {total_f/total_q:.3f}')

if __name__ == "__main__":

    perfrom_cbir(dataset_path = dataset_path, query_images_path = query_image_path,
                 distribution = distribution, proximity = proximity, channel_bins = channel_bins,
                 k = k, experimentation = experimentation, display_results = display_results)



