import os
from PIL import Image


#folders = os.listdir()
dir_path = "pictures/Chapters"

files_dir = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

'''
if not os.path.isdir("pictures/Chapters_backup"):
    os.mkdir("pictures/Chapters_backup")
'''
'''
for dir in files_dir:
    
    if not os.path.isdir(f"pictures/Chapters_backup/{dir}"):
        os.mkdir(f"pictures/Chapters_backup/{dir}")
    
    files = os.listdir(os.path.join(dir_path, dir))
    for file in files:
        if file.endswith(".png"):
            print("Converting image", file)
            filename = file.split(".")[0]
            path = os.path.join(dir_path, dir)
            image = Image.open(os.path.join(path, file))
            image = image.convert('RGB')
            image.save(f'{os.path.join(path, filename)}.webp', 'webp', optimize = True, quality = 50)
            #os.replace(os.path.join(path, file), "pictures/Chapters_backup/{dir}/{file}")
'''
other_dirs = ["pictures/Flags", "pictures/People"]

for dir in other_dirs:
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".png"):
            print("Converting image", file)
            filename = file.split(".")[0]
            image = Image.open(os.path.join(dir, file))
            image = image.convert('RGB')
            image.save(f'{os.path.join(dir, filename)}.webp', 'webp', optimize = True, quality = 50)
            #os.replace(os.path.join(path, file), "pictures/Chapters_backup/{dir}/{file}")
