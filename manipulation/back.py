from PIL import Image, ImageOps

def create_background(wall, floor, dimensions):
    """
    Create a background image with the wall on the upper half and the floor on the lower half.
    """
    wall_resized = wall.resize((dimensions[0], dimensions[1] // 2))
    floor_resized = floor.resize((dimensions[0], dimensions[1] // 2))
    background = Image.new('RGBA', dimensions)
    background.paste(wall_resized, (0, 0))
    background.paste(floor_resized, (0, dimensions[1] // 2))
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
