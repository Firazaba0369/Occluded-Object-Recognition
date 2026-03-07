import random

import cv2
import numpy as np
from pathlib import Path

def augment_occlusion(target_path, distractor_path, min_scale=0.5, max_scale=0.8):
    """
    Loads images with a specific filename prefix and extension using OpenCV
    Applies occlusion on the target using the distractor

    Args:
        target_path (str): path to one image occlusion will be applied to
        distractor_path (str): path to one image that will be used to occlude
        max_scale (int): minimum scale to transform distractor
        min_scale (int): maximum scale to transform distractor

    Returns:
        nothing, a image will be saved
    """

    rgb_images_target = [str(p) for p in Path(target_path).glob("*_crop.png")]
    depth_images_target = [str(p) for p in Path(target_path).glob("*_depthcrop.png")]

    rgb_images_distractor = [str(p) for p in Path(distractor_path).glob("*_crop.png")]
    depth_images_distractor = [str(p) for p in Path(distractor_path).glob("*_depthcrop.png")]
    mask_images_distractor = [str(p) for p in Path(distractor_path).glob("*_maskcrop.png")]

    # only one image for now for testing

    target_image_rgb = cv2.imread(rgb_images_target[0])
    target_image_depth = cv2.imread(depth_images_target[0], -1)

    distractor_image_rgb = cv2.imread(rgb_images_distractor[0])
    distractor_image_mask = cv2.imread(mask_images_distractor[0], 0)
    # in theory removes the white border around the cropped images since the masks weren't perfect.
    kernel = np.ones((5, 5), np.uint8)
    distractor_image_mask = cv2.erode(distractor_image_mask, kernel, iterations=1)

    distractor_image_depth = cv2.imread(depth_images_distractor[0], -1)

    # randomly resizing distractor so that it won't cover the entire target
    random_scale = random.uniform(min_scale, max_scale)
    resized_distractor_rgb = cv2.resize(distractor_image_rgb, None, fx=random_scale, fy=random_scale)
    resized_distractor_mask = cv2.resize(distractor_image_mask, None, fx=random_scale, fy=random_scale, interpolation=cv2.INTER_NEAREST)
    resized_distractor_depth = cv2.resize(distractor_image_depth, None, fx=random_scale, fy=random_scale)

    # inverting the distractor mask to use to subtract from target
    distractor_mask_inverted = cv2.bitwise_not(resized_distractor_mask)

    # gets the x, y sizes of the images
    target_height, target_width = target_image_rgb.shape[:2]
    distractor_height, distractor_width = resized_distractor_rgb.shape[:2]

    # using these coordinates for the random location on the target
    max_x = target_width - distractor_width
    max_y = target_height - distractor_height

    # if the distractor is somehow bigger than the target resize again with even smaller scale
    if max_x <= 0 or max_y <= 0:
        random_scale = random.uniform((min_scale - 0.2 if min_scale > 0.2 else 0), (max_scale - 0.2 if max_scale > 0.2 else 0.1))
        resized_distractor_rgb = cv2.resize(distractor_image_rgb, None, fx=random_scale, fy=random_scale)
        resized_distractor_mask = cv2.resize(distractor_image_mask, None, fx=random_scale, fy=random_scale, interpolation=cv2.INTER_NEAREST)
        resized_distractor_depth = cv2.resize(distractor_image_depth, None, fx=random_scale, fy=random_scale)

        # resetting all the previous coords with new resized results
        distractor_height, distractor_width = resized_distractor_rgb.shape[:2]
        max_x = target_width - distractor_width
        max_y = target_height - distractor_height
        distractor_mask_inverted = cv2.bitwise_not(resized_distractor_mask)

        # final check, at this point its a lost cause
        if max_x <= 0 or max_y <= 0:
            print("distractor is still too big. skipping augmentation.")
            return

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
    desired_distractor_depth = target_avg_depth - 50
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

    # writing the actual image
    cv2.imwrite("test_output_rgb.png", target_image_rgb)
    cv2.imwrite("test_output_depth.png", target_image_depth)

augment_occlusion(r"G:\Temp Storage for Final Project DL\rgbd-dataset\camera\camera_1", r"G:\Temp Storage for Final Project DL\rgbd-dataset\apple\apple_1")
