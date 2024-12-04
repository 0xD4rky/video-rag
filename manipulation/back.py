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

output_size = (1365, 768)
wall_height = int(output_size[1] * 0.7)
floor_height = output_size[1] - wall_height
wall_resized = wall_image.resize((output_size[0], wall_height))
floor_resized = floor_image.resize((output_size[0], floor_height))


final_background = Image.new("RGBA", output_size)
final_background.paste(wall_resized, (0, 0))
final_background.paste(floor_resized, (0, wall_height))

car_masked = apply_mask(car_image, mask)
car_position = (250, wall_height - car_masked.height // 4 + 50)
background_with_car = place_on_background(car_masked, final_background, car_position)

shadow_position = (car_position[0], car_position[1] + car_masked.height // 4)
final_image = add_shadow(background_with_car, shadow_mask, shadow_position)

final_image.show()
final_image.save("/Users/darky/Documents/cv_stack")
