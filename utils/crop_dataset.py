# Load modules
import os, cv2

import numpy as np
import pandas as pd

from pathlib import Path



# Save crop data into Excel with one sheet per subfolder
def save_crop_data_excel(crop_data, output_path):

    # Initialize a list to accumulate all the rows of data
    all_data = []

    # Loop through each subfolder and accumulate the data
    for subfolder, images in crop_data.items():

        for filename, coords in images.items():

            # Append each image's data along with its subfolder (sheet) information
            all_data.append([filename, subfolder] + coords)

    # Create a DataFrame from the accumulated data
    df = pd.DataFrame(all_data, columns = ['filename', 'subfolder', 'x_start', 'y_start', 'x_end', 'y_end'])
    
    # Write the data to Excel, each subfolder gets its own sheet
    with pd.ExcelWriter(output_path) as writer:

        for subfolder in crop_data.keys():

            # Filter the main DataFrame to get data only for the current subfolder
            subfolder_data = df[df['subfolder'] == subfolder]
            subfolder_data = subfolder_data.drop('subfolder', axis = 1)
            
            # Write the subfolder's data to the sheet in the Excel file
            subfolder_data.to_excel(writer, 
                                    sheet_name = subfolder[:31], 
                                    index = False
                                )


# Get the center of the `lumen` based on the convex hull of the non-zero region
def get_lumen_center(lumen):

    # Conver the mask to Numpy array
    lumen = np.array(lumen)
    height, width = lumen.shape

    # Find coordinates of non-zero mask pixels
    lumen_coords = np.column_stack(np.where(lumen > 0))
    
    # If there are no non-zero pixels, return None
    if len(lumen_coords) == 0:
        return (height // 2, width // 2), None    
    
    # Compute the convex hull of the non-zero mask pixels
    contours, _ = cv2.findContours(lumen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return (height // 2, width // 2), None
    
    if len(contours) > 1:
        print(f'Found {len(contours)} for the convex hull.')

    all_points = np.concatenate(contours)
    convex_hull = cv2.convexHull(all_points)
    
    # Compute the centroid of the convex hull (average of points in the hull)
    M = cv2.moments(convex_hull)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cY, cX), convex_hull
    
    else:

        # If the area is zero (corresponding to a degenerate hull), return the center of the mask directly
        return (np.mean(lumen_coords[:, 0]), np.mean(lumen_coords[:, 1])), convex_hull



# Crop the 2D or 3D image and mask to the specified crop size based on the `lumen` center
def crop_images(raw_image,
                lumen_image, 
                mask_image, 
                mdc_image,
                plaque_image,
                crop_size):

    # Get shapes
    height, width = raw_image.shape
    crop_height, crop_width = crop_size

    # Get the center of the `lumen` using the convex hull
    center, convex_hull = get_lumen_center(lumen_image)    
    lumen_center_y, lumen_center_x = center

    # Crop along height and width based on the center of the `lumen`
    y_start = max(lumen_center_y - crop_height // 2, 0)
    x_start = max(lumen_center_x - crop_width // 2, 0)

    # Ensure that the crop doesn't go out of bounds
    y_end = y_start + crop_height
    x_end = x_start + crop_width

    if y_end > height:
        y_start = max(height - crop_height, 0)
        y_end = height
    if x_end > width:
        x_start = max(width - crop_width, 0)
        x_end = width

    crop_coords = [x_start, y_start, x_end, y_end]

    cropped_image = raw_image[y_start:y_end, x_start:x_end]
    cropped_lumen = lumen_image[y_start:y_end, x_start:x_end]
    cropped_mask = mask_image[y_start:y_end, x_start:x_end]
    cropped_mdc = mdc_image[y_start:y_end, x_start:x_end]
    cropped_plaque = plaque_image[y_start:y_end, x_start:x_end]

    return (cropped_image, 
            cropped_lumen,
            cropped_mask, 
            cropped_mdc,
            cropped_plaque,
            crop_coords, 
            convex_hull, 
            center)


# Function which crops the data from all the subfolders
def process_folder(input_dir, output_dir, crop_size):

    # Initialize empty dictionary for the crop coordinates
    crop_data = dict()

    # Walk through the directory and its subfolders
    subfolders = [
                    f for f in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, f))
                 ]
    
    for fld in subfolders:    

        root = os.path.join(input_dir, fld)
        parent_folder = os.path.basename(root)
        
        # Find the corresponding subfolders
        raw_folder = os.path.join(root, 'png_raw')
        mask_folder = os.path.join(root, 'png_mask')
        mdc_folder = os.path.join(root, 'png_mdc')
        lumen_folder = os.path.join(root, 'png_lumen')
        plaque_folder = os.path.join(root, 'png_plaque')
        
        # Get all filenames in the raw folder
        filenames = os.listdir(raw_folder)
        
        # Loop over the filenames belonging to the current subfolder
        for raw_img_filename in filenames:
            
            print('\n')
            print(f'Processing [{raw_img_filename}]...')

            if raw_img_filename.endswith('.png'):

                mask_filename = raw_img_filename.replace('raw', 'mask')
                mdc_filename = raw_img_filename.replace('raw', 'mdc')
                lumen_filename = raw_img_filename.replace('raw', 'lumen')
                plaque_filename = raw_img_filename.replace('raw', 'plaque')

                # Read the image and mask files (assuming they have the same name)
                raw_image = cv2.imread(os.path.join(raw_folder, raw_img_filename), cv2.IMREAD_UNCHANGED)
                mdc_image = cv2.imread(os.path.join(mdc_folder, mdc_filename), cv2.IMREAD_UNCHANGED)
                
                mask_image = cv2.imread(os.path.join(mask_folder, mask_filename), cv2.IMREAD_GRAYSCALE)
                lumen_image = cv2.imread(os.path.join(lumen_folder, lumen_filename), cv2.IMREAD_GRAYSCALE)
                plaque_image = cv2.imread(os.path.join(plaque_folder, plaque_filename), cv2.IMREAD_GRAYSCALE)

                # Crop the images
                (cropped_raw, 
                 cropped_lumen,
                 cropped_mask, 
                 cropped_mdc,
                 cropped_plaque, 
                 crop_coords, 
                 convex_hull,
                 center) = crop_images(raw_image,
                                       lumen_image, 
                                       mask_image, 
                                       mdc_image,
                                       plaque_image, 
                                       crop_size)
                
                # Recreate the same folder structure in the output directory
                output_subfolder = os.path.join(output_dir, parent_folder)

                os.makedirs(os.path.join(output_subfolder, 'png_raw'), exist_ok = True)
                os.makedirs(os.path.join(output_subfolder, 'png_mask'), exist_ok = True)
                os.makedirs(os.path.join(output_subfolder, 'png_mdc'), exist_ok = True)
                os.makedirs(os.path.join(output_subfolder, 'png_lumen'), exist_ok = True)
                os.makedirs(os.path.join(output_subfolder, 'png_plaque'), exist_ok = True)
                os.makedirs(os.path.join(output_subfolder, 'png_lumen_crop'), exist_ok = True)

                # Save the cropped images in the corresponding subfolder
                cv2.imwrite(os.path.join(output_subfolder, f'png_raw/{raw_img_filename}'), cropped_raw)
                cv2.imwrite(os.path.join(output_subfolder, f'png_mask/{mask_filename}'), cropped_mask)
                cv2.imwrite(os.path.join(output_subfolder, f'png_mdc/{mdc_filename}'), cropped_mdc)
                cv2.imwrite(os.path.join(output_subfolder, f'png_lumen/{lumen_filename}'), cropped_lumen)
                cv2.imwrite(os.path.join(output_subfolder, f'png_plaque/{plaque_filename}'), cropped_plaque)

                # Create a version of the original `lumen` with crop rectangle
                lumen_with_box = cv2.cvtColor(lumen_image.copy(), cv2.COLOR_GRAY2BGR)
                x_start, y_start, x_end, y_end = crop_coords
                cv2.rectangle(lumen_with_box, (x_start, y_start), (x_end - 1, y_end - 1), color = (0, 255, 0), thickness = 2)

                if convex_hull is not None:
                    cv2.drawContours(lumen_with_box, [convex_hull], -1, (255, 0, 0), 2)

                if center is not None:
                    (cY, cX) = center
                    cv2.circle(lumen_with_box, (cX, cY), radius = 4, color = (0, 0, 255), thickness = 2)
                    
                cv2.imwrite(os.path.join(output_subfolder, f'png_lumen_crop/{lumen_filename}'), lumen_with_box)
                
                # Append the crop coordinates to the dictionary
                name = Path(raw_img_filename.replace('raw', '')).stem

                if fld not in crop_data:
                    crop_data[fld] = dict()
                
                crop_data[fld][name] = crop_coords

    return crop_data


# Main function
if __name__ == "__main__":

    crop_size = (128, 128)
    data_type = 'test'          # Choose between `train_val` and `test`
    
    input_dir = os.path.join("./Vascular_pathologies_data/", data_type)

    output_dir = f"Vascular_pathologies_data_cropped_{crop_size[0]}_{crop_size[1]}"
    os.makedirs(output_dir, exist_ok = True)

    output_dir = os.path.join(output_dir, data_type)
    os.makedirs(output_dir, exist_ok = True)

    crop_data = process_folder(input_dir = input_dir, 
                               output_dir = output_dir, 
                               crop_size = crop_size
                        )

    save_crop_data_excel(crop_data, os.path.join(output_dir, 'crop_coordinates.xlsx'))
