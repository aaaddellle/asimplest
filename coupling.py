import os
from PIL import Image
import itertools

def save_paired_data(spectrogram_path, image_path, save_path, img_size=(128, 128)):
    species_dirs = os.listdir(spectrogram_path) 
    for species in species_dirs:
        spec_dir = os.path.join(spectrogram_path, species)
        img_dir = os.path.join(image_path, species)

        
        species_save_dir = os.path.join(save_path, species)
        os.makedirs(species_save_dir, exist_ok=True)
        
        spec_files = sorted(os.listdir(spec_dir))
        img_files = sorted(os.listdir(img_dir))

        
        img_files = list(itertools.islice(itertools.cycle(img_files), len(spec_files)))

        for i, (spec_file, img_file) in enumerate(zip(spec_files, img_files)):
            
            spec_img = Image.open(os.path.join(spec_dir, spec_file)).convert('L')
            bird_img = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                    
            spec_img = spec_img.resize(img_size)
            bird_img = bird_img.resize(img_size)
            
            spec_img.save(os.path.join(species_save_dir, f"{i:04d}_spectrogram.png"))
            bird_img.save(os.path.join(species_save_dir, f"{i:04d}_image.png"))

def main():
    spectrogram_dir = 'C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\mels_1'
    image_dir = 'C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\images'
    save_dir = 'C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\paired_data'


    save_paired_data(spectrogram_dir, image_dir, save_dir)


if __name__ == '__main__':
    main()