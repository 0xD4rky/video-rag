from PIL import Image, ImageOps

def apply_mask(image, mask):
    """
    Applies a mask to an image to remove the background, retaining transparency.
    """
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)

def center_and_compose_background(car, wall, floor, output_size):
    """
    Composes the wall and floor into a single background image and centers the car on it.
    """
    wall_height = int(output_size[1] * 0.7)
    floor_height = output_size[1] - wall_height

    wall = wall.resize((output_size[0], wall_height))
    floor = floor.resize((output_size[0], floor_height))

    background = Image.new("RGBA", output_size)
    background.paste(wall, (0, 0))
    background.paste(floor, (0, wall_height))

    # Calculate the position to center the car
    car_position = ((output_size[0] - car.width) // 2, wall_height - (car.height // 2))
    background.paste(car, car_position, car)
    
    return background, car_position

def add_shadow(background, shadow_mask, car_position, car_size):
    """
    Adds a shadow under the car.
    """
    shadow_mask = shadow_mask.resize(car_size, Image.Resampling.LANCZOS)
    shadow = Image.new("RGBA", car_size, (0, 0, 0, 100))  # Adjust shadow transparency as needed
    shadow_position = (car_position[0], car_position[1] + (car_size[1] // 2))
    background.paste(shadow, shadow_position, shadow_mask)
    return background



config = {
    "car_path" : "/Users/darky/Documents/cv_stack/assignment/images/1.jpeg",
    "mask_path" : "/Users/darky/Documents/cv_stack/assignment/car_masks/1.png",
    "shadow_path" : "/Users/darky/Documents/cv_stack/assignment/shadow_masks/1.png",
    "wall_path" : "/Users/darky/Documents/cv_stack/assignment/wall.png",
    "floor_path" : "/Users/darky/Documents/cv_stack/assignment/floor.png"
}

car_image = Image.open(config["car_path"]).convert("RGBA")
mask = Image.open(config["mask_path"]).convert("L")
shadow_mask = Image.open(config["shadow_path"]).convert("L")
wall_image = Image.open(config["wall_path"]).convert("RGBA")
floor_image = Image.open(config["floor_path"]).convert("RGBA")

car_masked = apply_mask(car_image, mask)
background = compose_background(wall_image, floor_image, (1365, 768))
car_position = (250, 180)  # Adjust based on visual alignment
background_with_car = background.copy()
background_with_car.paste(car_masked, car_position, car_masked)

# Adjust shadow
shadow_size = (background_with_car.width, car_masked.height // 4)
shadow_position = (car_position[0], car_position[1] + car_masked.height - shadow_size[1])
final_image = add_shadow(background_with_car, shadow_mask, shadow_position, shadow_size)

# Save or display the final image
final_image.show()
