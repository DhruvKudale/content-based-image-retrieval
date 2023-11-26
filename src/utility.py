import os

def load_images_from_folder(folder_path):
    images = []
    labels = []
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                images.append(img_path)
                labels.append(subfolder_name)
    return images, labels


def perform_eval(query_image_path, images, labels, results):
    label_dict = {}
    actual_label = query_image_path.split('/')[-2:][0]
    assert len(labels) == len(images)
    total_gts = 0
    total_det = len(results)
    for i in range(len(labels)):
        label_dict[images[i]] = labels[i]
        if labels[i] == actual_label:
            total_gts = total_gts + 1
    tp = 0
    for entry in results:
        image_name = entry['image-name']
        detected_label = image_name.split('/')[-2:][0]
        if detected_label == actual_label:
            tp = tp + 1

    p = tp / total_det if total_det else 0
    r = tp / total_gts if total_gts else 0
    f = 2 * p * r / (p + r) if (p > 0 and r > 0) else 0
    return p, r, f