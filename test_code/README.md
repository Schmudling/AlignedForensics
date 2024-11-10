# Test Code

We use the testing pipeline provided by [CLIPDet](https://github.com/grip-unina/ClipBased-SyntheticImageDetection). We test on a wide variety of real and synthetic images. We make the dataset public here. 

## Evaluation Procedure 

To evaluate our models on the datsets, the path containing the image should be converted into a csv file. We provide a script in order to do that,

'''
python create_csv.py --base_folder "path to image folder" --output_csv "output csv path" --dir "real/fake (optional)"
'''


