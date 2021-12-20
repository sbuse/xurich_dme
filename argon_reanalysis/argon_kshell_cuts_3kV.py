
import awkward1 as ak
import numpy as np

def argon_kshell_cuts_3kV(events):
    #4) largest_s2_amplitude_pmt < 11600
    events= events[ak.all(events.s2.height_mvdc_bot <11600, axis=1)]
    
    #5)second_largest_s2_area_pmt = 0
    events= events[events.s2s_per_waveform == 1]
    
    #6)largest_s1_area_pmt >0 
    events = events[ak.any(events.s1.area_pe_bot>0,axis=1)]
    
    #7)largest_s2_area_pmt > largest_s1_area_pmt
    event = events[ak.max(events.s2.area_pe_bot,axis=1)>ak.max(events.s1.area_pe_bot,axis=1)]
    
    #8) 0.20 < s2_frt < 0.34
    events["s2","s2_frt"] = events.s2.area_pe_top/(events.s2.area_pe_top+events.s2.area_pe_bot)
    mask= ak.any(events.s2.s2_frt>0.16, axis=1)&ak.any(events.s2.s2_frt<0.34, axis=1)
    events=events[mask]
    
    #9)s2_area+s1_area>1175
    mask = ak.max(events.s2.area_pe_top,axis=1)+ak.max(events.s2.area_pe_bot,axis=1)+ak.max(events.s1.area_pe_top,axis=1)+\
    ak.max(events.s1.area_pe_bot,axis=1)>1175
    events=events[mask]
    
    #print("Length array pos 1")
    #print(ak.num(events,axis=0))
    if ak.num(events,axis=0)==0:
        return ak.Array([])
            
    
    #Add the drifttime
    mask_s1_before_s2 = events.s1.pos_bot < ak.flatten(events.s2.pos_bot)
    
    #Cut away the event which don't have an s1 befor the s2. these are s2 only.
    events = events[ak.any(events.s1.area_pe_bot[mask_s1_before_s2],axis=1)]
    
    #print("Length array pos 2")
    #print(ak.num(events,axis=0))
    if ak.num(events,axis=0)==0:
        return ak.Array([])

    mask_s1_before_s2 = events.s1.pos_bot < ak.flatten(events.s2.pos_bot)
    max_s1_before_s2 = ak.argmax(events.s1.area_pe_bot[mask_s1_before_s2],axis=1,keepdims=True)
    
    events["mask_max_s1_before_s2"] = max_s1_before_s2
    events["drifttime_musec"] = ak.flatten((ak.max(events.s2.pos_bot,axis=1)-events.s1.pos_bot[max_s1_before_s2])/100)
    
    #10) S2 width cut
    lower_width = 34.5+0.22*events["drifttime_musec"] + np.sqrt(32.6*events["drifttime_musec"])
    upper_width = 48.2+0.19*events["drifttime_musec"] + np.sqrt(47.3*events["drifttime_musec"])
    
    upper_mask = ak.any(events.s2.width_bot<upper_width,axis=1)
    lower_mask = ak.any(events.s2.width_bot>lower_width,axis=1)
    
    events = events[upper_mask&lower_mask]
    
    #11)FDV cut
    v = 1.9608556663165941
    gate_time = 1.537109944449019

    events["z"] = -v*(events["drifttime_musec"]-gate_time)
    events = events[events.z<-2]
    events = events[events.z>-28]
    
    events["r_s2"]= ak.flatten(np.sqrt(np.square(events.s2.x_corr)+np.square(events.s2.y_corr)))
    events = events[events.r_s2<10]
    
    return events
    
    
