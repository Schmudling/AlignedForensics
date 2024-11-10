import os
import csv

def create_csv_from_folder(base_folder, output_csv, dir=''):
    data = []

    for root, dirs, files in os.walk(base_folder):
        for file in sorted(files)[:3000]:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp') or file.endswith('.jpeg')or file.endswith('.JPEG'):  # Assuming we're only interested in PNG files
                file_path = os.path.join(root, file)
                file_type = file_path.split(os.sep)[1]  # Extracting the type from the folder structure
                data.append((file_path, dir))
        if len(data) > 3000:
            break

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'typ'])
        csvwriter.writerows(data)

# Usage
base_folder = '/nobackup/anirudh/datasets/evaluations/sd_pure/1_fake'
output_csv = 'data/sdpure_fake.csv'
create_csv_from_folder(base_folder, output_csv, dir='fake')
