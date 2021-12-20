#!/bin/bash

folder="3.00kV_BG"
nfiles=65
test=false

data_path=/disk/bulk_atp/thiemek/XurichII/processed_new/Run3/
saving_path=/disk/data8/student/sbuse/megatrees/

#data_path=/disk/bulk_atp/thiemek/XurichII/processed_without_splitting/Run3/
#saving_path=/disk/data8/student/sbuse/megatrees_without_splitting/

minitree_data_path=/disk/bulk_atp/thiemek/XurichII/minitrees/Run3/$folder/
file="files_to_process_$folder.txt"

create the lists of files to process: files_to_process_{folder}.txt
python3 2_list_processed_files.py $folder $nfiles $data_path $minitree_data_path $saving_path

part1="files_to_process_"
part2=".txt"

cat $part1$folder$part2 | while read line; do
  echo started $line $folder $test
  sbatch 3_launch_megatree.slurm $line $folder $test $saving_path
done
