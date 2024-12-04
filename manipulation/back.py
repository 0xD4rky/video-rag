from PIL import Image, ImageOps

def apply_mask(image, mask):
    """
    Apply a mask to an image, removing the background.
    :param image: PIL Image in RGBA.
    :param mask: PIL Image in grayscale (L mode) serving as the mask.
    :return: Image with the mask applied.
    """
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)

def place_on_background(car, background, position):
    """
    Place the car image on the specified background at the given position.
    :param car: PIL Image of the car (with transparency).
    :param background: PIL Image of the background.
    :param position: Tuple (x, y) for where to place the car.
    :return: Background with the car placed on it.
    """
    background.paste(car, position, car)
    return background

def add_shadow(background, shadow_mask, position):
    """
    Add shadow to the car, enhancing the 3D effect.
    :param background: PIL Image of the background with the car.
    :param shadow_mask: PIL Image, grayscale mask for the shadow.
    :param position: Tuple (x, y) for where to place the shadow.
    :return: Background with the shadow added.
    """
    shadow = Image.new("RGBA", shadow_mask.size, (0, 0, 0, 150))  # Shadow color and opacity
    shadow = shadow.resize(background.size, Image.Resampling.LANCZOS)
    background.paste(shadow, position, shadow_mask)
    return background

car_image = Image.open("/path/to/car_image.jpeg").convert("RGBA")
mask = Image.open("/path/to/mask.png").convert("L")
shadow_mask = Image.open("/path/to/shadow_mask.png").convert("L")
wall_image = Image.open("/path/to/wall_image.png").convert("RGBA")
floor_image = Image.open("/path/to/floor_image.png").convert("RGBA")

congif = {
    "car_path" : "/path/to/car_image.jpeg",
    "mask_path" : "/path/to/mask.png",
    "shadow_path" : "/path/to/shadow_mask.png",
    "wall_path" : "/path/to/wall_image.png",
    "floor_path" : "/path/to/floor_image.png"
}

output_size = (1365, 768)
wall_height = int(output_size[1] * 0.7)
floor_height = output_size[1] - wall_height
wall_resized = wall_image.resize((output_size[0], wall_height))
floor_resized = floor_image.resize((output_size[0], floor_height))


final_background = Image.new("RGBA", output_size)
final_background.paste(wall_resized, (0, 0))
final_background.paste(floor_resized, (0, wall_height))

