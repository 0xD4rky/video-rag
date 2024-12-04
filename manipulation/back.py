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

def scale_and_center_image(image, background, scale_factor, vertical_offset=0, horizontal_offset=0):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    center_x = (background.width - new_width) // 2 + horizontal_offset
    center_y = (background.height - new_height) // 2 + vertical_offset
    return resized_image, (center_x, center_y)

def place_car_and_shadow(background, car, car_mask, shadow_mask):
    car_scaled, (car_x, car_y) = scale_and_center_image(car, background, 1.75, vertical_offset=250)
    car_mask_scaled, _ = scale_and_center_image(car_mask, background, 1.75)
    car_mask_array = np.array(car_mask_scaled)
    binary_car_mask = np.where(car_mask_array > 128, 255, 0).astype(np.uint8)
    car_masked = Image.composite(car_scaled, Image.new("RGBA", car_scaled.size), Image.fromarray(binary_car_mask, 'L'))

    shadow_scaled, (shadow_x, shadow_y) = scale_and_center_image(shadow_mask, background, 1.75, vertical_offset=625)
    shadow_alpha = np.array(shadow_scaled.convert('L'))
    shadow_alpha = np.where(shadow_alpha > 128, 128, 0).astype(np.uint8)
    shadow_rgba = np.stack([shadow_alpha]*3 + [shadow_alpha], axis=-1)
    shadow_image = Image.fromarray(shadow_rgba, 'RGBA')

    background.paste(car_masked, (car_x, car_y), car_masked)
    background.paste(shadow_image, (shadow_x + 35, shadow_y), shadow_image)

    return background


wall_image = Image.open("/Users/darky/Documents/cv_stack/assignment/wall.png").convert("RGBA")
floor_image = Image.open("/Users/darky/Documents/cv_stack/assignment/floor.png").convert("RGBA")
car_image = Image.open("/Users/darky/Documents/cv_stack/assignment/images/3.jpeg").convert("RGBA")
car_mask_image = Image.open("/Users/darky/Documents/cv_stack/assignment/car_masks/3.png").convert("L")
shadow_mask_image = Image.open("/Users/darky/Documents/cv_stack/assignment/shadow_masks/3.png").convert("RGBA")

background_image = remove_black_mask_and_join(wall_image,floor_image)  

final_image = place_car_and_shadow(background_image, car_image, car_mask_image, shadow_mask_image)

final_image.show()
