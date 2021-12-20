# change form v15 to v16. Additional info, the number of pulses that the pmt saw. Proper height conversion to mVdc. Removing unused data.  18.05.2021 Simon Buse

import numpy as np
import sys,os
import time
import awkward as ak
import uproot 
import ROOT
import numba as nb
import tracemalloc
import gc


def correct_xy(x_uncor,y_uncor):
    #this is the fixed version after the studing the x-y position reconstr. 
    c=6.5
    x_scaled= x_uncor/c
    y_scaled= y_uncor/c

    r_uncor = np.sqrt(x_uncor**2+y_uncor**2)
    if r_uncor >=7:
        return x_uncor*(1+(15.5-r_uncor)/r_uncor),y_uncor*(1+(15.5-r_uncor)/r_uncor)
        
    x_mapped = c*x_scaled*np.sqrt(1-y_scaled**2/2)
    y_mapped = c*y_scaled*np.sqrt(1-x_scaled**2/2)

    r = np.sqrt(x_mapped**2+y_mapped**2)
    a = (-31.0/np.pi*np.arccos(r/5.8)+15.5) if r/5.8 <= 1 else 15.5

    x_corrected = a/r*x_mapped
    y_corrected = a/r*y_mapped
    
    return x_corrected,y_corrected

@nb.jit(nopython=True)
def nb_wsum(array,weights):
    #superfast weighted sum
    weighted_sum=0.0
    sum_of_weights=0.0

    for i,j in zip(array,weights):
        weighted_sum += i*j
        sum_of_weights += j
    return weighted_sum/sum_of_weights

def undo_trigger_overflow(ttt):
    #creates a continous trigger time tag with zero at the start of file. 
    ttt = np.array(ttt)
    ttt = ttt%(2**31-1)
    jumps = [0]
    normed_ttt=ttt/(2**31-1)

    for i in range(len(normed_ttt)):
        if i!= len(normed_ttt)-1 and normed_ttt[i]>normed_ttt[i+1]:
            jumps.append(i+1)

    jumps.append(len(normed_ttt))

    summed_ttt=np.array([])
    count = 1
    for j in range(len(jumps)):
        if j+1<len(jumps):
            summed_ttt= np.append(summed_ttt,count*(2**31-1)+ttt[jumps[j]:jumps[j+1]])
            count +=1
        if j==len(jumps):
            summed_ttt= np.append(summed_ttt,count*(2**31-1)+ttt[jumps[j]:])

    return summed_ttt


def create_megatree(processed_file,folder):    
    print("-----start processing file: {}-----".format(processed_file))

    #first read the triggertimetags, and find the timing information by undoing the overflow
    ttt_file = "/disk/bulk_atp/thiemek/XurichII/processed_new/Run3/TTT/"+processed_file[-16:-5]+"_TTT.root"
    print("-----reading the TriggerTimeTag file: {}-----".format(ttt_file))
    ttt = uproot.open(ttt_file+":t1").arrays()
    ttt["corr_ttt"] = undo_trigger_overflow(ttt["TriggerTimeTag"])
    
    #create a new ROOT file and add the Branches.
    print("---------create a new file in {}-------------".format(saving_path+folder+"/"))
    file = ROOT.TFile(saving_path+folder+"/"+"megatree_{}.root".format(processed_file[-16:-5]), 'recreate')
    megatree = ROOT.TTree("t1", "a pretty megatree")

    #s2_pos_top       = np.zeros(Nmax*1, dtype=float)
    s2_pos_bot       = np.zeros(Nmax*1, dtype=float)
    s2_area_top      = np.zeros(Nmax*1, dtype=float)
    s2_area_bot      = np.zeros(Nmax*1, dtype=float)
    #s2_width_top     = np.zeros(Nmax*1, dtype=float)
    s2_width_bot     = np.zeros(Nmax*1, dtype=float)
    s2_width10_bot   = np.zeros(Nmax*1, dtype=float)
    #s2_height_top    = np.zeros(Nmax*1, dtype=float)
    s2_height_bot    = np.zeros(Nmax*1, dtype=float)
    s2_x_uncorr      = np.zeros(Nmax*1, dtype=float)
    s2_y_uncorr      = np.zeros(Nmax*1, dtype=float)
    s2_x_corr        = np.zeros(Nmax*1, dtype=float)
    s2_y_corr        = np.zeros(Nmax*1, dtype=float)
    s2_pulse_time    = np.zeros(Nmax*1, dtype=float)

    s1_pos_top       = np.zeros(Nmax*1, dtype=float)
    s1_pos_bot       = np.zeros(Nmax*1, dtype=float)
    s1_area_top      = np.zeros(Nmax*1, dtype=float)
    s1_area_bot      = np.zeros(Nmax*1, dtype=float)
    #s1_width_top     = np.zeros(Nmax*1, dtype=float)
    s1_width_bot     = np.zeros(Nmax*1, dtype=float)
    s1_width10_bot   = np.zeros(Nmax*1, dtype=float)
    #s1_height_top    = np.zeros(Nmax*1, dtype=float)
    s1_height_bot    = np.zeros(Nmax*1, dtype=float)
    s1_pulse_time    = np.zeros(Nmax*1, dtype=float)
       
    s2s_per_waveform = np.zeros(1, dtype=int)
    s1s_per_waveform = np.zeros(1, dtype=int)
    wave_number      = np.zeros(1, dtype=int)
    unix_time        = np.zeros(1, dtype=float)
    trigger_time_tag_uncorr = np.zeros(1, dtype=float)
    trigger_time_tag_corr   = np.zeros(1, dtype=float)
    sat_bools        = np.zeros(len(ttt["corr_ttt"]), dtype=bool)  
    preceding_is_sat = np.zeros(1, dtype=bool)
    npulses_pmt_stor = np.zeros(len(ttt["corr_ttt"]), dtype=int)
    npulses_pmt      = np.zeros(1, dtype=int)

    megatree.Branch("s2s_per_waveform",s2s_per_waveform,"s2s_per_waveform/I")
    megatree.Branch("s1s_per_waveform",s1s_per_waveform,"s1s_per_waveform/I")
    megatree.Branch("wave_number",wave_number,"wave_number/I")
    megatree.Branch("unix_time",unix_time, "unix_time/D")
    megatree.Branch("trigger_time_tag_uncorr",trigger_time_tag_uncorr, "trigger_time_tag_uncorr/D")
    megatree.Branch("trigger_time_tag_corr",trigger_time_tag_corr, "trigger_time_tag_corr/D")
    megatree.Branch("preceding_is_sat",preceding_is_sat, "preceding_is_sat/O")
    megatree.Branch("npulses_pmt",npulses_pmt,"npulses_pmt/I")

    #megatree.Branch("s2_pos_top",s2_pos_top,"s2_pos_top[s2s_per_waveform]/D")
    megatree.Branch("s2_pos_bot",s2_pos_bot,"s2_pos_bot[s2s_per_waveform]/D")
    megatree.Branch("s2_area_pe_top",s2_area_top,"s2_area_pe_top[s2s_per_waveform]/D")
    megatree.Branch("s2_area_pe_bot",s2_area_bot,"s2_area_pe_bot[s2s_per_waveform]/D")
    #megatree.Branch("s2_width_top",s2_width_top,"s2_width_top[s2s_per_waveform]/D")
    megatree.Branch("s2_width_bot",s2_width_bot,"s2_width_bot[s2s_per_waveform]/D")
    megatree.Branch("s2_width10_bot",s2_width10_bot,"s2_width10_bot[s2s_per_waveform]/D")
    #megatree.Branch("s2_height_mvdc_top",s2_height_top,"s2_height_mvdc_top[s2s_per_waveform]/D")
    megatree.Branch("s2_height_mvdc_bot",s2_height_bot,"s2_height_mvdc_bot[s2s_per_waveform]/D")
    megatree.Branch("s2_x_uncorr",s2_x_uncorr,"s2_x_uncorr[s2s_per_waveform]/D")
    megatree.Branch("s2_y_uncorr",s2_y_uncorr,"s2_y_uncorr[s2s_per_waveform]/D")
    megatree.Branch("s2_x_corr",s2_x_corr,"s2_x_corr[s2s_per_waveform]/D")
    megatree.Branch("s2_y_corr",s2_y_corr,"s2_y_corr[s2s_per_waveform]/D")
    megatree.Branch("s2_pulse_time",s2_pulse_time,"s2_pulse_time[s2s_per_waveform]/D")

    #megatree.Branch("s1_pos_top",s1_pos_top,"s1_pos_top[s1s_per_waveform]/D") 
    megatree.Branch("s1_pos_bot",s1_pos_bot,"s1_pos_bot[s1s_per_waveform]/D")
    megatree.Branch("s1_area_pe_top",s1_area_top, "s1_area_pe_top[s1s_per_waveform]/D")
    megatree.Branch("s1_area_pe_bot",s1_area_bot, "s1_area_pe_bot[s1s_per_waveform]/D")
    #megatree.Branch("s1_width_top",s1_width_top, "s1_width_top[s1s_per_waveform]/D")
    megatree.Branch("s1_width_bot",  s1_width_bot, "s1_width_bot[s1s_per_waveform]/D")
    megatree.Branch("s1_width10_bot",s1_width_bot, "s1_width10_bot[s1s_per_waveform]/D")
    #megatree.Branch("s1_height_mvdc_top",s1_height_top, "s1_height_mvdc_top[s1s_per_waveform]/D")
    megatree.Branch("s1_height_mvdc_bot",s1_height_bot, "s1_height_mvdc_bot[s1s_per_waveform]/D")
    megatree.Branch("s1_pulse_time",s1_pulse_time,"s1_pulse_time[s1s_per_waveform]/D")
    
    

    tree_count=0
    variables =["Time","EventNumber","PeakID","PeakArea","PeakWidth","PeakWidth10","PeakHeight","PeakPosition",
            "PulsesSensorNo","PulsesHitNo","PulsesPosition"]

    print("---start scanning wavenforms---")
    for tree in uproot.iterate(processed_file+":T1", filter_name=variables,step_size=load_size,library="np"):

        PulsesSensorNo_tree = [i.tolist() for i in tree["PulsesSensorNo"]]
        PulsesHitNo_tree    = [i.tolist() for i in tree["PulsesHitNo"]]
        PeakArea_tree       = [i.tolist() for i in tree["PeakArea"]]
        PeakPosition_tree   = [i.tolist() for i in tree["PeakPosition"]]
        PeakWidth_tree      = [i.tolist() for i in tree["PeakWidth"]]
        PeakWidth10_tree    = [i.tolist() for i in tree["PeakWidth10"]]
        PeakHeight_tree     = [i.tolist() for i in tree["PeakHeight"]]
        PeakID_tree         = [i.tolist() for i in tree["PeakID"]]           
        
        
        for waveform in range(len(PeakID_tree)):
            
                        #skipp waveform if there is no s2 in it or a saturated signal.
            #PeakID_tree[waveform] = [[1,1,2,0,1],[1,2]....[]], starting with the pmt sublist
            if np.any(np.array(PeakHeight_tree[waveform][0])>11600):
                sat_bools[waveform+tree_count*load_size] = True
                continue
            
            if 2 not in PeakID_tree[waveform][0]: 
                continue

            s2_count  = 0
            s1_count  = 0
            ttt_corr   = ttt["corr_ttt"][waveform+tree_count*load_size]
            ttt_uncorr = ttt["TriggerTimeTag"][waveform]

            PulsesSensorNo_list = PulsesSensorNo_tree[waveform]
            PulsesHitNo_list    = PulsesHitNo_tree[waveform]
            PeakArea_list       = PeakArea_tree[waveform]
            PeakPosition_list   = PeakPosition_tree[waveform]
            PeakWidth_list      = PeakWidth_tree[waveform]
            PeakWidth10_list    = PeakWidth10_tree[waveform]
            PeakHeight_list     = PeakHeight_tree[waveform]
            PeakID_list         = PeakID_tree[waveform]
            npulses_pmt_stor[waveform+tree_count*load_size]  = len(PeakID_list[0])
            
            #------------ Search for S2 in the pulses in each waveform, Attention they are not filtered for PMT+1SiPM----------
            #PeakID_tree[waveform] = [[1,1,2,0,1],[1,2]....[]], starting with the pmt sublist
            for pulse_sensors,pulse_hit_pos in zip(PulsesSensorNo_list,PulsesHitNo_list):
                
                #there must be a PMT signal + Sipm, only PMT or only sipm does not count. 
                if 0 in pulse_sensors and len(np.unique(pulse_sensors))>=2:

                    peak_id = PeakID_list[0][pulse_hit_pos[pulse_sensors.index(0)]]

                    #Compute S2 variables if PMT+ 1SiPM only save them up to Nmax entries.
                    if peak_id ==2 and s2_count<Nmax:
                        
                        #Load the S2 variables if there is a candidate pulse
                        leng=len(pulse_sensors)-1

                        s2_area_sipm         = np.zeros((leng,), np.float64)
                        s2_pos_sipm          = np.zeros((leng,), np.float64)
                        #s2_width_sipm        = np.zeros((leng,), np.float64)
                        #s2_height_sipm       = np.zeros((leng,), np.float64)
                        pulse_sensors_sipm   = np.zeros((leng,), np.int64)

                        count_nonzero = 0
                        for i,j in zip(pulse_sensors,pulse_hit_pos):
                            if i==0:
                                s2_area_pmt     = PeakArea_list[i][j]/gains[0]*conversion
                                s2_pos_pmt      = PeakPosition_list[i][j]
                                s2_width_pmt    = PeakWidth_list[i][j]
                                s2_width10_pmt  = PeakWidth10_list[i][j]
                                s2_height_pmt   = PeakHeight_list[i][j]*2.25/(2**14)*1e3
                               
                            elif PeakArea_list[i][j]>0:
                                s2_area_sipm[count_nonzero]   = PeakArea_list[i][j]/gains[i]*conversion
                                #s2_pos_sipm[count_nonzero]    = PeakPosition_list[i][j]
                                #s2_width_sipm[count_nonzero]  = PeakWidth10_list[i][j]
                                #s2_height_sipm[count_nonzero] = PeakHeight_list[i][j]
                                pulse_sensors_sipm[count_nonzero]   = i
                                count_nonzero +=1
                        
                        #Top weights
                        #Compute the sipm area=weight, needed for all other variables
                        w_t      = s2_area_sipm
                        sum_w_t  = np.sum(w_t)

                        #when sum_w_t =0, then all the s2_areas where negative. 
                        if sum_w_t ==0:
                            continue

                        
                        #Compute the S2 variables, 
                        s2_x_uncor_ = (np.sum(x_pos_sipm[pulse_sensors_sipm]*w_t))/sum_w_t
                        s2_y_uncor_ = (np.sum(y_pos_sipm[pulse_sensors_sipm]*w_t))/sum_w_t
                        s2_x_cor_,s2_y_cor_ = correct_xy(s2_x_uncor_,s2_y_uncor_)

                        #-----------Store variables in arrays---------------
                        #pulse_time, pmt is the reference
                        s2_pulse_time[s2_count] = ttt_corr-3000+s2_pos_pmt if s2_pos_pmt<3000 else ttt_corr+s2_pos_pmt

                        #Top variables
                        #Numba weighted sum for same length arrays is faster than np.average and ak.mean
                        #Attention same length is not checked!
                        s2_area_top[s2_count]   = sum_w_t
                        #s2_pos_top[s2_count]    = nb_wsum(s2_pos_sipm,w_t)
                        #s2_width_top[s2_count]  = nb_wsum(s2_width_sipm,w_t)
                        #s2_height_top[s2_count] = nb_wsum(s2_height_sipm,w_t)    

                        #Bottom variables
                        s2_area_bot[s2_count]   = s2_area_pmt
                        s2_pos_bot[s2_count]    = s2_pos_pmt
                        s2_width_bot[s2_count]  = s2_width_pmt
                        s2_width10_bot[s2_count]= s2_width10_pmt
                        s2_height_bot[s2_count] = s2_height_pmt

                        s2_x_uncorr[s2_count]   = s2_x_uncor_
                        s2_y_uncorr[s2_count]   = s2_y_uncor_              

                        s2_x_corr[s2_count]     = s2_x_cor_
                        s2_y_corr[s2_count]     = s2_y_cor_              
                        s2_count+=1


                    #------------Compute S1 variables if PMT+ 1SiPM only save them up to Nmax entries.-----------
                    elif peak_id ==1 and s1_count<Nmax:

                        #Load the S1 variables if there is a candidate pulse
                        leng=len(pulse_sensors)-1

                        s1_area_sipm         = np.zeros((leng,), np.float64)
                        #s1_pos_sipm          = np.zeros((leng,), np.float64)
                        #s1_width_sipm        = np.zeros((leng,), np.float64)
                        #s1_height_sipm       = np.zeros((leng,), np.float64)

                        count_nonzero = 0
                        for i,j in zip(pulse_sensors,pulse_hit_pos):
                            if i==0:
                                s1_area_pmt     = PeakArea_list[i][j]/gains[0]*conversion
                                s1_pos_pmt      = PeakPosition_list[i][j]
                                s1_width_pmt    = PeakWidth_list[i][j]
                                s1_width10_pmt  = PeakWidth10_list[i][j]
                                s1_height_pmt   = PeakHeight_list[i][j]*2.25/(2**14)*1e3
                                
                                
                            elif PeakArea_list[i][j]>0:
                                s1_area_sipm[count_nonzero]   = PeakArea_list[i][j]/gains[i]*conversion
                                #s1_pos_sipm[count_nonzero]    = PeakPosition_list[i][j]
                                #s1_width_sipm[count_nonzero]  = PeakWidth10_list[i][j]
                                #s1_height_sipm[count_nonzero] = PeakHeight_list[i][j]
                                count_nonzero +=1
                        
                        #Top weights are the sipm area, needed for all other variables
                        #Attention the area is transformed to pe within read variables
                        w_t      = s1_area_sipm
                        sum_w_t  = np.sum(w_t)
                        
                        #when sum_w_t =0, then all the s1_areas where negative. 
                        if sum_w_t ==0:
                            continue
                        
                        #-----------Store variables in arrays---------------
                        #pulse_time, pmt is the reference, the pos [0,6000] wrt to the ttt_corr                         
                        s1_pulse_time[s1_count] = ttt_corr-3000+s1_pos_pmt if s1_pos_pmt<3000 else ttt_corr+s1_pos_pmt 
                        
                        #Top variables
                        #Numba weighted sum for same length arrays is faster than np.average and ak.mean
                        #Attention same length is not checked!!
                        s1_area_top[s1_count]   = sum_w_t
                        #s1_pos_top[s1_count]    = nb_wsum(s1_pos_sipm,w_t)
                        #s1_width_top[s1_count]  = nb_wsum(s1_width_sipm,w_t)
                        #s1_height_top[s1_count] = nb_wsum(s1_height_sipm,w_t)  

                        #Bottom variables
                        s1_area_bot[s1_count]   = s1_area_pmt
                        s1_pos_bot[s1_count]    = s1_pos_pmt
                        s1_width_bot[s1_count]  = s1_width_pmt
                        s1_width10_bot[s1_count]= s1_width10_pmt
                        s1_height_bot[s1_count] = s1_height_pmt
                        s1_count+=1
                  
            if s2_count!=0:
                wave_number[0]            = tree["EventNumber"][waveform]
                unix_time[0]              = tree["Time"][waveform]
                s2s_per_waveform[0]       = int(s2_count)
                s1s_per_waveform[0]       = int(s1_count)
                trigger_time_tag_corr[0]  = ttt_corr
                trigger_time_tag_uncorr[0]= ttt_uncorr
                
                wave_index = waveform+tree_count*load_size
                if wave_index>=0:
                    preceding_is_sat[0]   = sat_bools[wave_index-1] 
                npulses_pmt[0] = npulses_pmt_stor[wave_index]
                
                megatree.Fill()            
        
        megatree.Write("",ROOT.TFile.kOverwrite)
        #megatree.AutoSave("FlushBaskets")
        #megatree.FlushBaskets()
               
        #Evaluate the execution time
        end = time.time()    
        m, s = divmod(end-start, 60)
        h, m = divmod(m, 60)
#        current, peak = tracemalloc.get_traced_memory()
#         print("scanned {} wavenforms in {:.0f}:{:0.0f}:{:0.0f}, currently using {} MB; peak was {} MB"\
#              .format((tree_count+1)*load_size,h, m, s,current/10**6,peak/10**6))
        print("scanned {} wavenforms in {:.0f}:{:0.0f}:{:0.0f}".format((tree_count+1)*load_size,h, m, s))

        tree_count += 1
        if test and tree_count==5:
            break
            
    file.Close()
    print("-----finished processing file: {}-----".format(processed_file))
    if not test:
        os.chdir(saving_path+folder+"/")
        os.rename(r"megatree_{}.root".format(processed_file[-16:-5]),r"megatree_complete_{}.root".format(processed_file[-16:-5]))

#static information
conversion = (2.25/2**14)/(50*1.60217662*10**(-19))*10*10**(-9)
# the zero is there to correct for sipm 1..16
x_pos_sipm = np.array([0,10.875,4.125,10.875,4.125,10.875,10.875,4.125,4.125,-10.875,-4.125,-10.875,-4.125,-10.875,-10.875,-4.125,-4.125])
y_pos_sipm = np.array([0,10.875,10.875,4.125,4.125,-4.125,-10.875,-4.125,-10.875,-10.875,-10.875,-4.125,-4.125,4.125,10.875,4.125,10.875])
gains      = np.array([3.76476e6,3.14135e7,3.22513e7,3.17634e7,3.221e7,3.08955e7,3.0359e7,3.08114e7,3.08897e7,3.0966e7,
                  3.15564e7,3.09457e7,3.12228e7,3.16169e7,3.1554e7,3.11147e7,3.18335e7])

pmt_gain_low   = 3.76476e6
pmt_gain_high  = 4.44865e6

#read input
processed_file = sys.argv[1]
folder         = sys.argv[2]
test           = sys.argv[3] =="true"
saving_path    = sys.argv[4]


#test settings
if test :
    print("this is a test run")
    load_size = 1000       #load this amount of waveforms per loop 


#run settings
Nmax = 50            #max number of s1 or s2 events per waveform to be saved
if not test:
    print("this is NOT a test run")
    load_size = 2000     #load this amount of waveforms per loop 


#Start programm
start = time.time()
#tracemalloc.start()

create_megatree(processed_file,folder)

#snapshot = tracemalloc.take_snapshot()
# for stat in snapshot.statistics("lineno")[:10]:
#     print(stat)

#tracemalloc.stop()
#Evaluate the execution time
end = time.time()    
m, s = divmod(end-start, 60)
h, m = divmod(m, 60)
print("It took: %d:%02d:%02d to execute this code. " % (h, m, s)) 
