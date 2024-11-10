MID_REAL='/nobackup3/anirudh/datasets/redcaps/samples'
MID_FAKE='/nobackup3/anirudh/datasets/SDv2-1/cifar-100/res-512/guidance-7-5/images/1_fake'
DIR='sweep_plots/res/'

if [ -d "$DIR" ]; then
  echo "Directory $DIR already exists."
else
  echo "Directory $DIR does not exist. Creating it now."
  mkdir -p "$DIR"
  echo "Directory $DIR created."
fi


python resize_sweep.py --real=$MID_REAL --fake=$MID_FAKE --start=128 --end=1024 --stride=250 --device 'cuda:2' --out_path=$DIR --base_res=512

#python resize_sweep.py --real=$MID_REAL --fake=$MID_FAKE --setting 'qual' --start=0 --end=100 --stride=25 --device 'cuda:5' --out_path=$DIR #--base_res=512
