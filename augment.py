import random
from pathlib import Path

import cv2
import numpy as np
import os
import shutil

def augment_occlusion(t_rgb_path, t_depth_path, d_rgb_path, d_depth_path, d_mask_path,
                      output_rgb_name, output_depth_name, output_path, min_scale=0.5, max_scale=0.8):
    """
    Loads images with a specific filename prefix and extension using OpenCV
    Applies occlusion on the target using the distractor

    Args:
        t_rgb_path (str): path to one image occlusion will be applied to
        t_depth_path (str): path to one image that will be used to occlude
        d_rgb_path (str): path to the distractor rgb image
        d_depth_path (str): path to the distractor depth image
        d_mask_path (str): path to the distractor mask image
        output_rgb_name (str): the output name for the rgb image w/ occlusion applied to it
        output_depth_name (str): the name for depth map with occlusion applied to it
        output_path (str): the path to a folder put the image in
        max_scale (float): minimum scale to transform distractor
        min_scale (float): maximum scale to transform distractor

    Returns:
        nothing, a image will be saved
    """

    target_image_rgb = cv2.imread(t_rgb_path)
    target_image_depth = cv2.imread(t_depth_path, -1)

    distractor_image_rgb = cv2.imread(d_rgb_path)
    distractor_image_mask = cv2.imread(d_mask_path, 0)
    # in theory removes the white border around the cropped images since the masks weren't perfect.
    kernel = np.ones((5, 5), np.uint8)
    distractor_image_mask = cv2.erode(distractor_image_mask, kernel, iterations=1)

    distractor_image_depth = cv2.imread(d_depth_path, -1)

    # get the original x, y sizes of the images
    target_height, target_width = target_image_rgb.shape[:2]
    distractor_height, distractor_width = distractor_image_rgb.shape[:2]

    # calculate what scale is required to make the distractor fit exactly inside the target
    scale_x = target_width / distractor_width
    scale_y = target_height / distractor_height

    # use the smaller of the two scales ensuring object fits without changing aspect ratio
    base_scale = min(scale_x, scale_y)

    # apply min/max scale as a percentage of the base scale
    random_scale = base_scale * random.uniform(min_scale, max_scale)

    # resize everything using this new scale
    resized_distractor_rgb = cv2.resize(distractor_image_rgb, None, fx=random_scale, fy=random_scale)
    resized_distractor_mask = cv2.resize(distractor_image_mask, None, fx=random_scale, fy=random_scale,
                                         interpolation=cv2.INTER_NEAREST)
    resized_distractor_depth = cv2.resize(distractor_image_depth, None, fx=random_scale, fy=random_scale)

    # invert the distractor mask
    distractor_mask_inverted = cv2.bitwise_not(resized_distractor_mask)

    # recalc dimensions for the ROI using the final resized images
    distractor_height, distractor_width = resized_distractor_rgb.shape[:2]

    # ensures that floating point rounding doesn't accidentally make max_x negative
    max_x = max(0, target_width - distractor_width)
    max_y = max(0, target_height - distractor_height)

    # random spot to put the distractor
    target_x = random.randint(0, max_x)
    target_y = random.randint(0, max_y)

    # creating the region of interest for where the distractor will go
    roi = target_image_rgb[target_y:target_y + distractor_height, target_x:target_x + distractor_width]
    roi_depth = target_image_depth[target_y:target_y + distractor_height, target_x:target_x + distractor_width]

    # figure out how far away target is in the roi
    valid_target_depths = roi_depth[roi_depth > 0]
    target_avg_depth = np.mean(valid_target_depths) if len(valid_target_depths) > 0 else 1000

    # figure out how far the distractor is
    valid_distractor_depths = resized_distractor_depth[resized_distractor_mask > 0]
    distractor_avg_depth = np.mean(valid_distractor_depths) if len(valid_distractor_depths) > 0 else 1000

    # shift the distractor so it is closer than the target
    random_depth_offset = random.randint(20, 150) # 20mm to 150mm
    desired_distractor_depth = target_avg_depth - random_depth_offset
    depth_shift = desired_distractor_depth - distractor_avg_depth

    # apply shift to only the actual object pixels, now the distractor is "in front" of the target in the depth map
    temp_depth = resized_distractor_depth.astype(np.float32)
    temp_depth[resized_distractor_mask > 0] += depth_shift
    temp_depth = np.clip(temp_depth, 1, 65535)  # prevent negative distances
    resized_distractor_depth = temp_depth.astype(np.uint16)

    # blacks out where the distractor will go
    roi_bg = cv2.bitwise_and(roi, roi, mask=distractor_mask_inverted)
    roi_bg_depth = cv2.bitwise_and(roi_depth, roi_depth, mask=distractor_mask_inverted)

    # grabs only the object from the distractor
    distractor_fg = cv2.bitwise_and(resized_distractor_rgb, resized_distractor_rgb, mask=resized_distractor_mask)
    distractor_depth_fg = cv2.bitwise_and(resized_distractor_depth, resized_distractor_depth, mask=resized_distractor_mask)

    # puts the distractor on top of the black void in the target, then adds it back to the image
    dst_rgb = cv2.add(roi_bg, distractor_fg)
    dst_depth = cv2.add(roi_bg_depth, distractor_depth_fg)

    # use the above to them back into the actual image
    target_image_rgb[target_y:target_y + distractor_height, target_x:target_x + distractor_width] = dst_rgb
    target_image_depth[target_y:target_y + distractor_height, target_x:target_x + distractor_width] = dst_depth

    # writing the actual image to specified folder
    os.makedirs(output_path, exist_ok=True)
    full_output_path_rgb = os.path.join(output_path, output_rgb_name)
    full_output_path_depth = os.path.join(output_path, output_depth_name)
    cv2.imwrite(full_output_path_rgb, target_image_rgb)
    cv2.imwrite(full_output_path_depth, target_image_depth)


def get_random_frame(dataset_root):
    """navigates Categories -> Instances -> Frames to get one random object.
    Assumes the dataset structure remained unchanged from when it was downloaded.

    Args:
        dataset_root: the root of the washington rgbd dataset

    Returns:
        The paths for the rgb, depth, and mask of that one random object
    """
    root_path = Path(dataset_root)
    categories = [d for d in root_path.iterdir() if d.is_dir()]

    while True:
        # pick a random category ('cereal_box')
        chosen_category = random.choice(categories)

        # pick a random instance ('cereal_box_1')
        instances = [d for d in chosen_category.iterdir() if d.is_dir()]
        chosen_instance = random.choice(instances)

        # pick a random frame ('cereal_box_1_1_24_crop.png')
        rgb_files = list(chosen_instance.glob("*_crop.png"))
        chosen_rgb = random.choice(rgb_files)

        # construct the matching Depth and Mask paths
        # strip "_crop.png" to get the base name ("cereal_box_1_1_24")
        base_name = chosen_rgb.name.replace("_crop.png", "")

        chosen_depth = chosen_instance / f"{base_name}_depthcrop.png"
        chosen_mask = chosen_instance / f"{base_name}_maskcrop.png"

        # some of the images don't have a matching depth/mask for some reason
        if chosen_depth.exists() and chosen_mask.exists():
            return str(chosen_rgb), str(chosen_depth), str(chosen_mask), chosen_category.name
        
def generate_masked_sets(dataset_root, min_scale, max_scale, output_folder, num_images):
    """
    Applies occlusion on a known set of images from the dataset with random images from the dataset
    Puts the resulting images into the output folder specified.

    Args:
        dataset_root: the root of the washington rgbd dataset
        min_scale (float): minimum scale of distractor transform
        max_scale (float): maximum scale of distractor transform
        output_folder (str): the name of the output folder
        num_images (int): number of images being created

    Returns:
        nothing, folders will be made
    """

    for image_number in range(num_images):
        random_target_rgb, random_target_depth, _, target_label = get_random_frame(dataset_root)
        random_distractor_rgb, random_distractor_depth, random_distractor_mask, distractor_label = get_random_frame(dataset_root)
            
        rgb_out = f"{target_label}_random_rgb_{image_number}.png"
        depth_out = f"{target_label}_random_depth_{image_number}.png"

        augment_occlusion(random_target_rgb, random_target_depth, random_distractor_rgb, random_distractor_depth,
                          random_distractor_mask, rgb_out, depth_out, output_folder, min_scale, max_scale)

    print("masked set generation complete")

def split_dataset(dataset_root, output_folder, train_ratio):
    """
    Splits the dataset into train and test sets by moving files into new folders

    Args:
        dataset_root: the root of the washington rgbd dataset
        output_folder: the name of the output folder
        train_ratio (float): percentage of images to be put in the training set

    Returns:
        nothing, folders will be made
    """

    # get root path and categories
    root_path = Path(dataset_root)
    output_path = Path(output_folder)
    categories = [d for d in root_path.iterdir() if d.is_dir()]

    # get all rgb and depth files and split them into train and test sets
    for category in categories:
        category_name = category.name
        instances = [d for d in category.iterdir() if d.is_dir()]
        rgb_files = []

        # only look at rgb files to determine train/test split, depth files will be moved with their matching rgb files
        for instance in instances:
            rgb_files.extend([
                f for f in instance.glob("*_crop.png")
                if "depthcrop" not in f.name
            ])

        # shuffle the files to ensure random distribution of train/test sets
        random.shuffle(rgb_files)
        num_train = int(len(rgb_files) * train_ratio)

        # split into train and test sets
        train_files = rgb_files[:num_train]
        test_files = rgb_files[num_train:]

        # move the files into their train and test folders
        for file in train_files:
            dest = output_path / "train" / category_name / file.parent.name
            dest.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file, dest / file.name)

            depth_file = file.parent / file.name.replace("_crop.png", "_depthcrop.png")
            if depth_file.exists():
                shutil.copy2(depth_file, dest / depth_file.name)

            mask_file = file.parent / file.name.replace("_crop.png", "_maskcrop.png")
            if mask_file.exists():
                shutil.copy2(mask_file, dest / mask_file.name)

        for file in test_files:
            dest = output_path / "test" / category_name / file.parent.name
            dest.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file, dest / file.name)

            depth_file = file.parent / file.name.replace("_crop.png", "_depthcrop.png")
            if depth_file.exists():
                shutil.copy2(depth_file, dest / depth_file.name)

            mask_file = file.parent / file.name.replace("_crop.png", "_maskcrop.png")
            if mask_file.exists():
                shutil.copy2(mask_file, dest / mask_file.name)

    print("dataset splitting complete")

split_dataset(r".\rgbd-dataset", r".\rgbd-dataset_split", 0.7)
generate_masked_sets(r".\rgbd-dataset_split\train", 0.1, 0.9, r".\masked_sets\train", 500)
generate_masked_sets(r".\rgbd-dataset_split\test", 0.2, 0.4, r".\masked_sets\test\low_occlusion", 200)
generate_masked_sets(r".\rgbd-dataset_split\test", 0.4, 0.6, r".\masked_sets\test\medium_occlusion", 200)
generate_masked_sets(r".\rgbd-dataset_split\test", 0.6, 0.8, r".\masked_sets\test\high_occlusion", 200)