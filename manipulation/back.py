from PIL import Image
import numpy as np

def remove_black_mask_and_join(wall, floor):
    """
    Removes the black mask ([255, 255, 255, 0]) from both images and joins them vertically.
    """
    wall_array = np.array(wall)
    floor_array = np.array(floor)

    # Identifying non-mask areas (pixels not equal to [255, 255, 255, 0])
    wall_mask = ~(np.all(wall_array == [255, 255, 255, 0], axis=-1))
    floor_mask = ~(np.all(floor_array == [255, 255, 255, 0], axis=-1))

    # Apply mask and keep dimensions
    wall_filtered = wall_array[wall_mask].reshape((-1, wall_array.shape[1], 4))
    floor_filtered = floor_array[floor_mask].reshape((-1, floor_array.shape[1], 4))

    # Combine the images
    combined_array = np.vstack((wall_filtered, floor_filtered))
    combined_image = Image.fromarray(combined_array)
    return combined_image

def scale_and_center_image(image, background, scale_factor):
    """
    Scale and center the image on the background.
    """
    # Calculate the new size
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate center positions
    center_x = (background.width - new_width) // 2
    center_y = (background.height - new_height) // 2
    
    return resized_image, (center_x, center_y)


def place_car_and_shadow(background, car, car_mask, shadow_mask):
    """
    Place the car using the car mask and align the shadow using the shadow mask on the background.
    """
    # Convert the images to arrays for manipulation
    car_array = np.array(car)
    car_mask_array = np.array(car_mask)
    shadow_mask_array = np.array(shadow_mask.convert('L'))  # Ensure shadow mask is in grayscale

    # Mask the car and prepare for placement
    car_masked_array = np.where(car_mask_array[:, :, None] == 255, car_array, 0)

    # Convert shadow mask to binary (shadow/non-shadow)
    shadow_threshold = 128  # Adjust this threshold based on your shadow mask's specifics
    shadow_binary_mask = np.where(shadow_mask_array > shadow_threshold, 255, 0).astype(np.uint8)

    # Resize the shadow to match the car size
    shadow_resized_mask = Image.fromarray(shadow_binary_mask).resize(car.size, Image.LANCZOS)
    shadow_resized_array = np.array(shadow_resized_mask)

    # Create an RGBA shadow image (semi-transparent black)
    shadow_rgba = np.zeros((shadow_resized_array.shape[0], shadow_resized_array.shape[1], 4), dtype=np.uint8)
    shadow_rgba[:, :, 3] = shadow_resized_array  # Alpha channel from shadow mask
    shadow_image = Image.fromarray(shadow_rgba, 'RGBA')

    # Calculate positions for the car and shadow
    car_x = (background.width - car.width) // 2
    car_y = (background.height - car.height) // 2

    # Place the car using its mask
    background.paste(Image.fromarray(car_masked_array, 'RGBA'), (car_x, car_y), Image.fromarray(car_mask_array, 'L'))

    # Adjust shadow placement relative to the car
    shadow_x = car_x
    shadow_y = car_y + car.height - 20  # Shift the shadow slightly to simulate realistic lighting
    background.paste(shadow_image, (shadow_x, shadow_y), shadow_image)

    return background

# Load and process images
wall_image = Image.open("/Users/darky/Documents/cv_stack/assignment/wall.png").convert("RGBA")
floor_image = Image.open("/Users/darky/Documents/cv_stack/assignment/floor.png").convert("RGBA")
car_image = Image.open("/Users/darky/Documents/cv_stack/assignment/images/1.jpeg").convert("RGBA")
car_mask_image = Image.open("/Users/darky/Documents/cv_stack/assignment/car_masks/1.png").convert("L")
shadow_mask_image = Image.open("/Users/darky/Documents/cv_stack/assignment/shadow_masks/1.png").convert("RGBA")

# Combine wall and floor images
background_image = remove_black_mask_and_join(wall_image,floor_image)  

# Place car and shadow on the background
final_image = place_car_and_shadow(background_image, car_image, car_mask_image, shadow_mask_image)

# Display and save the final image
final_image.show()
# final_image.save("final_output.png")
