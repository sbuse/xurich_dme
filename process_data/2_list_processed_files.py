import os,sys

#---------read input--------------
folder = sys.argv[1]
nfiles = int(sys.argv[2])
processed_data_path = sys.argv[3]
minitree_data_path  = sys.argv[4]
saving_path         = sys.argv[5]


#-------check which files are already processed---------- 
complete_megatrees_id = []
for entry in os.scandir(saving_path+folder+"/"):
    if (entry.path.startswith(saving_path+folder+"/megatree_complete_")):
        complete_megatrees_id.append(entry.path[-16:-5])

        
#-------list the files which need to be processed, check which files are in the minitree folders--------
total_files_in_folder=[]
list_of_files =[]
for entry in os.scandir(minitree_data_path):
    if (entry.path.endswith(".root")):
        if int(entry.path[-16:-10]) > 190524:
            total_files_in_folder.append(entry.path)
        if int(entry.path[-16:-10]) > 190524 and entry.path[-16:-5] not in complete_megatrees_id and len(list_of_files)<nfiles:
            list_of_files.append(processed_data_path+"processed_"+entry.path[-16:]+"\n")


#------write them into a txt files ----------------
with open('files_to_process_{}.txt'.format(folder), 'w+') as f:
    f.writelines(list_of_files)

print("In folder "+str(folder)+" are "+str(len(total_files_in_folder))+" files.")
print("{} of them are on the list to be processed".format(len(list_of_files)))
print("For {}: {} megatrees are completely processed.".format(folder,len(complete_megatrees_id))) 
        


  