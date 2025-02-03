# Test Code

We use the testing pipeline provided by [CLIPDet](https://github.com/grip-unina/ClipBased-SyntheticImageDetection). We test on a wide variety of real and synthetic images. We make the dataset public on huggingface at this [link](https://huggingface.co/datasets/AniSundar18/Robust_LDM_Benchmark). 

## Evaluation Procedure 
To run inference with the models, we need to refer to their path in the config file. An example of one such config file is present in the ```weights``` folder. The config file needs to be edited such that, file_path stores the path to the model weights.

To evaluate our models on the datsets, the path containing the image should be converted into a csv file. We provide a script in order to do that,

```
python create_csv.py --base_folder "path to image folder" --output_csv "output csv path" --dir "real/fake (optional)"
```

Once the datasets are ready, the per-image scores can be calculated as follows,
```
python main.py --in_csv {images csv file} --out_csv {output csv file} --device 'cuda:4' --weights_dir {weights directory} --models {models to evalute}
```
In order to run these evaluations parallely, use the ```run_tests.py``` script. The returned csv files contains the scores given by the neural networks. 

## Resolution/Compression sweeps
We also provide the code to recreate the resolution/webp-compression diagrams given in the paper. Examples can be found in the ```sweep.sh``` script. 


## Citation
If you find this code useful in your research, consider citing our work:
```
@misc{rajan2024effectivenessdatasetalignmentfake,
      title={On the Effectiveness of Dataset Alignment for Fake Image Detection}, 
      author={Anirudh Sundara Rajan and Utkarsh Ojha and Jedidiah Schloesser and Yong Jae Lee},
      year={2024},
      eprint={2410.11835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.11835}, 
}
```
