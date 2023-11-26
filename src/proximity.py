import cv2
import numpy as np

def quantize_pixel(pixel, n_colours):
    k = int(n_colours ** (1 / 3))
    factor = int(256 / k)
    return [int(pixel[0] / factor), int(pixel[1] / factor), int(pixel[2] / factor)]

def get_color_number_from_pixel(pixel, n_colours):
    k = int(n_colours ** (1 / 3))
    return k * k * pixel[0] + k * pixel[1] + pixel[2]

def convert_quantized_image_to_visualize(quantized_img, n_colours):
    k = int(n_colours ** (1 / 3))
    factor = int(256 / k)
    final_image = []
    for pixel in quantized_img.reshape(-1, 3):
        final_image.append([int(pixel[0] * factor), int(pixel[1] * factor), int(pixel[2] * factor)])
    final_image_np = np.asarray(final_image)
    final_image_np = final_image_np.reshape(quantized_img.shape)
    return final_image_np

def get_preprocessed_image(image, n_colours):
    # Blur the image
    blurred_image = cv2.blur(image, (3, 3))
    quantized_image = []
    for pixel in blurred_image.reshape(-1, 3):
        quantized_image.append(quantize_pixel(pixel, n_colours))
    quantized_image_np = np.asarray(quantized_image)
    quantized_image_np = quantized_image_np.reshape(image.shape)
    return quantized_image_np

def get_colour_mask_from_quantized_image(quantized_image, n_colours):
    color_mask = []
    for pixel in quantized_image.reshape(-1, 3):
        color_mask.append(get_color_number_from_pixel(pixel, n_colours))
    return np.asarray(color_mask)

def get_histo_masks(image):
    height, width, c = image.shape

    # Define the region to keep (central 75%)
    h_start = int(0.125 * height)
    h_end = int(0.875 * height)
    w_start = int(0.125 * width)
    w_end = int(0.875 * width)

    # Create a binary mask
    central_mask = np.zeros((height, width), dtype=np.uint8)
    central_mask[h_start: h_end, w_start: w_end] = 255
    peripheral_mask = ~central_mask
    return central_mask, peripheral_mask

def compare_basic_histograms(img1, img2, bins):
    # Calculate histograms
    channel_bins = [bins, bins, bins]
    ranges = [0, 256, 0, 256, 0, 256]
    hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, channel_bins, ranges)
    hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, channel_bins, ranges)

    # Normalize histograms
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Distances
    corr = np.corrcoef(hist_img1.ravel(), hist_img2.ravel())[0, 1]
    l1 = np.sum(np.abs(hist_img1.ravel() - hist_img2.ravel()))
    l2 = np.sqrt(np.sum((hist_img1.ravel() - hist_img2.ravel()) ** 2))
    return np.abs(corr), l1, l2

def compare_split_histograms(img1, img2, bins):
    # Calculate histograms
    channel_bins = [bins, bins, bins]
    ranges = [0, 256, 0, 256, 0, 256]

    # Get masks
    central_mask_1, peripheral_mask_1 = get_histo_masks(img1)
    central_mask_2, peripheral_mask_2 = get_histo_masks(img2)

    # Compute Central Histograms
    central_hist_img1 = cv2.calcHist([img1], [0, 1, 2], central_mask_1, channel_bins, ranges)
    central_hist_img2 = cv2.calcHist([img2], [0, 1, 2], central_mask_2, channel_bins, ranges)

    # Compute Peripheral Histograms
    peripheral_hist_img1 = cv2.calcHist([img1], [0, 1, 2], peripheral_mask_1, channel_bins, ranges)
    peripheral_hist_img2 = cv2.calcHist([img2], [0, 1, 2], peripheral_mask_2, channel_bins, ranges)

    # Normalize histograms
    cv2.normalize(central_hist_img1, central_hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(central_hist_img2, central_hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(peripheral_hist_img1, peripheral_hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(peripheral_hist_img2, peripheral_hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Distances
    corr = np.corrcoef(central_hist_img1.ravel(), central_hist_img2.ravel())[0, 1] + \
           np.corrcoef(peripheral_hist_img1.ravel(), peripheral_hist_img2.ravel())[0, 1]
    l1 = np.sum(np.abs(central_hist_img1.ravel() - central_hist_img2.ravel())) + np.sum(
        np.abs(peripheral_hist_img1.ravel() - peripheral_hist_img2.ravel()))
    l2 = np.sqrt(np.sum((central_hist_img1.ravel() - central_hist_img2.ravel()) ** 2)) + np.sqrt(
        np.sum((peripheral_hist_img1.ravel() - peripheral_hist_img2.ravel()) ** 2))
    return np.abs(corr / 2), l1 / 2, l2 / 2

### Color Coherence Vectors

def get_ccv_from_quantized_image(image, channel_bins, tau):
    n_colours = channel_bins ** 3
    quantized_image = get_preprocessed_image(image, n_colours)
    # Initialize coherence vector
    ccv_vector = []
    # Get color mask from quantized image
    colour_mask = get_colour_mask_from_quantized_image(quantized_image, n_colours)
    shaped_colour_mask = colour_mask.reshape(quantized_image.shape[0], quantized_image.shape[1])

    # Iterate over each discretized color
    for i in range(n_colours):
        # Extract pixels with the current color
        binary_mask = (shaped_colour_mask == i).astype(np.uint8)
        # Find connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)
        # Initialize coherence pair
        alpha_j, beta_j = 0, 0
        # Iterate over connected components
        for j in range(1, stats.shape[0]):
            component_size = stats[j, cv2.CC_STAT_AREA]
            # Classify pixels as coherent or incoherent based on the size
            if component_size > tau:
                alpha_j += component_size
            else:
                beta_j += component_size
        # Add coherence pair to the vector
        ccv_vector.append(alpha_j)
        ccv_vector.append(beta_j)
    return np.asarray(ccv_vector)
def compare_ccvs(img1, img2, channel_bins, tau):
    ccv1 = get_ccv_from_quantized_image(img1, channel_bins, tau)
    ccv2 = get_ccv_from_quantized_image(img2, channel_bins, tau)
    # Normalize CCVs
    ccv1 = (ccv1 - np.min(ccv1)) / (np.max(ccv1) - np.min(ccv1))
    ccv2 = (ccv2 - np.min(ccv2)) / (np.max(ccv2) - np.min(ccv2))
    l1 = np.sum(np.abs(ccv1 - ccv2))
    l2 = np.sqrt(np.sum((ccv1 - ccv2) ** 2))
    corr = np.corrcoef(ccv1, ccv2)[0, 1]
    return np.abs(corr), l1, l2

