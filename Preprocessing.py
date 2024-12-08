import glob
from PIL import Image
import numpy as np
import os

def summary_statistics(root_dir):
    """
    Across all images contained in root_dir: calculates minimum, average, and maximum widths and heights.
    Result is returned as a dictionary, where each width, height pair is a tuple
    ***** order of tuple is width, height *****
    """

    # list all folders (classes)
    image_widths  = []
    image_heights = []
    all_items = glob.glob('*', root_dir= root_dir)
    folders = [folder for folder in all_items if folder.find('.') == -1]

    # iteratively add all image widths and heights to respective lists
    for folder in folders:
        target_directory = root_dir + '/' + folder + '/'
        list_of_images = glob.glob('*', root_dir=target_directory)
        for image in list_of_images:
            src = target_directory + '/' + image
            w, h = Image.open(src).size
            image_widths.append(w)
            image_heights.append(h)

    # calculate min, mean, max of width and height; pair up each in tuple
    image_widths = np.array(image_widths)
    image_heights = np.array(image_heights)
    min_width_height = (image_widths.min(), image_heights.min())
    avg_width_height = (image_widths.mean(), image_heights.mean())
    max_width_height = (image_widths.max(), image_heights.max())

    # package statistics together into a dictionary; return
    stats = {'min dimensions' : min_width_height, \
             'avg dimensions' : avg_width_height, 'max dimensions' : max_width_height}
    return stats


def crop_image(image_path, target_width, target_height):
    """
    Crop image located at image_path to specified width and height, and return the image.
    ***I chose to crop the center of the image (rather than i.e., the bottom left corner). This gave the best chance
    for the subject of the image to be in the final crop. If you crop the bottom left corner, you usually just get the photo's
    background.

    https://www.geeksforgeeks.org/python-pil-image-crop-method/
    """

    # find original image dimensions
    image = Image.open(image_path)
    orig_image_width, orig_image_height = image.size

    # crop() requires us to provide the coordinates of the crop as a tuple as (left, upper, right,lower)
    # calculate left and right coordinates
    if (orig_image_width > target_width):
        center = float(orig_image_width) / 2.0
        left = center - (float(target_width) / 2.0)
        right = left + target_width
        #print("width:  center is " + str(center) + ", left is " + str(left) + ", right is " + str(right))
    else:
        left = 0
        right = target_width

    # calculate upper and lower coordinates
    if (orig_image_height > target_height):
        center = float(orig_image_height) / 2.0
        upper = center - (float(target_height) / 2.0)
        lower = upper + target_height
        #print("height:  center is " + str(center) + ", lower is " + str(lower) + ", upper is " + str(upper))
    else:
        lower = target_height
        upper = 0

    # crop and return image
    coordinates = (246.5,247.5 , 344.5,  343.5)
    new_image = image.crop(coordinates)
    return new_image


def process_dataset(orig_dataset_path, processing_function, dest_dataset_path, new_width, new_height):
    """
    processes images in dataset, based on processing_function passed in (i.e., passing crop_image calls crop_image on all images)
    Size of output images are specified by params new_width and new_height
    Processed images are saved to new directly specified by dest_dataset_path
    """
    # create new directory and duplicate its subfolder structure
    os.mkdir(dest_dataset_path)
    orig_all_items = glob.glob('*', root_dir= orig_dataset_path)
    orig_folders = [folder for folder in orig_all_items if folder.find('.') == -1]
    for folder in orig_folders:
        subfolder = dest_dataset_path + folder + '/'
        os.mkdir(subfolder)

    # iterate through subfolders of original dataset, process the image using processing_function, and save it to the equivalent
    # subfolder in dest_dataset_path
    new_all_items = glob.glob('*', root_dir=dest_dataset_path)
    new_folders = [folder for folder in new_all_items if folder.find('.') == -1]
    for orig_folder in orig_folders:
        orig_subfolder = orig_dataset_path + orig_folder + '/'
        orig_images = glob.glob('*.jpg', root_dir=orig_subfolder)
        for orig_image in orig_images:
            orig_image_path = orig_subfolder + orig_image
            new_image_object = processing_function(orig_image_path, new_width, new_height)
            new_image_path = dest_dataset_path + orig_folder + "/cropped_" + orig_image
            new_image_object.save(new_image_path)

# creates the cropped dataset
dataset_statistics = summary_statistics('dataset/')
process_dataset('dataset/', crop_image, 'dataset_cropped/', dataset_statistics['min dimensions'][0], dataset_statistics['min dimensions'][1])
