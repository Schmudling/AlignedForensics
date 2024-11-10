# Test Code

We use the testing pipeline provided by [CLIPDet](https://github.com/grip-unina/ClipBased-SyntheticImageDetection). We test on a wide variety of real and synthetic images. We make the dataset public here. 

## Evaluation Procedure 
The models need to be saved in a specific way, please refer to the ```weights``` folder for an example,

To evaluate our models on the datsets, the path containing the image should be converted into a csv file. We provide a script in order to do that,

```
python create_csv.py --base_folder "path to image folder" --output_csv "output csv path" --dir "real/fake (optional)"
```

Once the datasets are ready, the per-image scores can be calculated as follows,
```
python main.py --in_csv {images csv file} --out_csv {output csv file} --device 'cuda:4' --weights_dir {weights directory} --models {models to evalute}
```
In order to run these evaluations parallely, use the ```run_tests.py``` script.
