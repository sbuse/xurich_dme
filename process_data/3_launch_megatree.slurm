#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=1000
#SBATCH -o output/create_megatree-%j.txt

DATADIR=/home/uzh/sbuse/1_MT/experimental_rate/new_minitrees/batch_all/
cd $DATADIR

python3 -u 4_create_megatreev16.py $1 $2 $3 $4  #pass in processed_file,folder,test,saving_path

#No scratch required since creating_megatrees_v3.py writes directly into /disk/data8/student/sbuse/megatrees/