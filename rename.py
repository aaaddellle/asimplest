import os

base_dir = 'asimplest\images'  # Change this to the path where your categories are stored
categories = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

for category in categories:
    category_path = os.path.join(base_dir, category)
    images = os.listdir(category_path)
    images.sort()  # Optional: Sort files by name or modify to sort by creation time etc.

    # Optional: Create a mapping file to track old and new names
    with open(os.path.join(category_path, 'mapping.txt'), 'w') as file:
        for idx, image in enumerate(images):
            new_name = f'image_{idx+1:03d}.jpg'
            old_path = os.path.join(category_path, image)
            new_path = os.path.join(category_path, new_name)
            os.rename(old_path, new_path)

            file.write(f'{image} -> {new_name}\n')  # Log the change to a file

            print(f'Renamed {image} to {new_name}')
