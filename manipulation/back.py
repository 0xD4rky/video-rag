from PIL import Image, ImageOps

def apply_mask(image, mask):
    """
    Apply a mask to remove the background from the car image.
    """
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)

def resize_image(image, target_size):
    """
    Resize image maintaining aspect ratio.
    """
    return ImageOps.contain(image, target_size)

def compose_background(wall, floor, dimensions):
    """
    Compose the wall and floor into a single background image.
    """
    wall_height = int(dimensions[1] * 0.7)
    floor_height = dimensions[1] - wall_height
    wall = wall.resize((dimensions[0], wall_height))
    floor = floor.resize((dimensions[0], floor_height))
    
    background = Image.new("RGBA", dimensions)
    background.paste(wall, (0, 0))
    background.paste(floor, (0, wall_height))
    return background

def add_shadow(background, shadow_mask, position, size):
    """
    Add shadow to the image, adjusting size and position.
    """
    shadow_mask = shadow_mask.resize(size, Image.Resampling.LANCZOS)
    shadow = Image.new("RGBA", size, (0, 0, 0, 150))
    background.paste(shadow, position, shadow_mask)
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
