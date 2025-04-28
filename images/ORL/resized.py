import cv2
import os

# Path utama ke folder (ganti kalau beda)
base_folder = 'images/ORL' # contoh: '/Users/rubi/Documents/gambar'

# Path folder output
output_base = os.path.join(base_folder, 'resized')

# Target ukuran
target_width = 92
target_height = 112

# Bikin folder 'resized' kalau belum ada
if not os.path.exists(output_base):
    os.makedirs(output_base)

# Proses folder p39 sampai p50
for i in range(43, 44):
    folder_name = f'p{i}'
    folder_path = os.path.join(base_folder, folder_name)

    # Cek folder asal
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan, skip.")
        continue

    # Bikin subfolder di 'resized' sesuai folder asal
    output_folder = os.path.join(output_base, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Proses file 1.jpg sampai 10.jpg
    for j in range(1, 11):
        img_filename = f'{j}.jpg'
        img_path = os.path.join(folder_path, img_filename)

        # Load gambar
        img = cv2.imread(img_path)
        if img is None:
            print(f"Gagal baca gambar: {img_path}, skip.")
            continue

        # Resize
        resized_img = cv2.resize(img, (target_width, target_height))

        # Save ke folder 'resized'
        output_img_path = os.path.join(output_folder, img_filename)
        cv2.imwrite(output_img_path, resized_img)

        print(f"Berhasil resize dan simpan: {output_img_path}")

print("Semua gambar selesai diproses!")
