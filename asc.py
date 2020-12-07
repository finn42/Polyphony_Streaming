# auditory streaming complexity functions

import music21 as m21
import copy
import sys
import time
import math
import numpy as np 
import scipy as sp
import pandas as pd


def Onset_List(m21stream):
    # this function returns the timepoints of note onsets (in score time, beats)
    # reduced to a list of unique timepoints
    allOns = []
    # first: is this a score?   
    if isinstance(m21stream,m21.stream.Score):  
        # retreive all note onsets from score
        for p in m21stream:
            if isinstance(p,m21.stream.Part):
                ons = Onset_List(p)
                allOns.extend(ons)
    if isinstance(m21stream,m21.stream.Part):
        flattenedp = m21stream.flat.notesAndRests
        rest_state = 0
        tie_state = 0
        for el in flattenedp:
            if isinstance(el,m21.note.Note):
                rest_state = 0
                if el.tie:
                    if el.tie.type == 'start':
                        allOns.append(float(el.offset))
                else:
                    allOns.append(float(el.offset))
            if isinstance(el,m21.note.Rest):
                if rest_state == 0:
                    # allOns.append(float(el.offset)) # Do not include rest onsets
                    rest_state = 1
        allOns.append(float(el.offset+el.quarterLength)) # end of last note or rest     
    # extract unique time values
    ons = np.unique(allOns)
    return ons

def Diatonic_Number(note):
    # translates a music21 note element to it's number in diatonic values, 
    # instead of chromatic like midi
    diatonic = {'C':0,'D':1,'E':2,'F':3,'G':4,'A':5,'B':6}
    dPitch = diatonic[note.name[0]] + 7 + note.octave*7
    return dPitch

def Onset_Pitch_List(m21part):
    # this function returns the timepoints and pitch of note onsets (in score time, beats) for parts
    # counts notes only as their diatonic numbers, not midi chromatic values
    # forcing el.offset and el.quarterlength to floats for indexing purposes
    ons = [[],[]]
    # input must be a part
    if isinstance(m21part,m21.stream.Part):    
        # retreive all note onsets from score
        flattenedp = m21part.flat.notesAndRests
        rest_state = 0
        for el in flattenedp:
            if isinstance(el,m21.note.Note):
                rest_state = 0
                if el.tie:
                        if el.tie.type == 'start':
                            ons[0].append(float(el.offset))
                            #ons[1].append(el.pitch.ps)
                            ons[1].append(Diatonic_Number(el))
                else:
                    ons[0].append(float(el.offset))
                    #ons[1].append(el.pitch.ps)
                    ons[1].append(Diatonic_Number(el))
            if isinstance(el,m21.note.Rest):
                 if rest_state == 0:
                    ons[0].append(float(el.offset)) 
                    ons[1].append(math.nan)
                    rest_state = 1
        ons[0].append(float(el.offset+el.quarterLength))
        ons[1].append(math.nan)
    return ons

def beat_Times(score):
    # generate complete beatTimes list for interpolated pitch sequences and melodies
    onsAll = Onset_List(score)
    step = np.min(np.diff(onsAll))
    if np.min(np.diff(onsAll))<1:
        step = np.min([np.min(np.diff(onsAll)),1-np.min(np.diff(onsAll))])
    rate_options = [8,6,4,3,2,1,0.5,0.25,0.125] # heavy handed fix of the rates to covenient beat values
    close_rate =np.argmin(abs(rate_options -(1/step)))
    new_step = 1/rate_options[close_rate]
    beatTime = np.arange(onsAll[0],onsAll[-1]+new_step,new_step)
    return beatTime    

def Pitch_Beat_Interpolate(onsets,pitchValues,beatTime):
    # function to produce pitch sequence at times beatTime from onset list with corresponding pitch values
    # apparently this kind of interpolation isn't an option in pandas?!
    # first align min values
    if onsets[0]>beatTime[0]:
        onsets = beatTime[0] + onsets
        pitchValues = math.nan + pitchValues
    beatTime_pitch = np.array(np.zeros_like(beatTime), dtype=np.float)
    onsets_h = np.array(onsets)
    for i_beats in range(len(beatTime)):
        r = np.max(np.where(onsets_h<=beatTime[i_beats]))
        beatTime_pitch[i_beats] = pitchValues[np.max(np.where(onsets_h<=beatTime[i_beats]))]
    return beatTime_pitch

def Score_Pitch_Steps(score):
    # from an m21 score, evaluate the pitch squence in each part and outputs
    # panda dataframe with colums of current pitch values for even sampling (on smallest subdivision)
    df_Score = pd.DataFrame()
    if isinstance(score,m21.stream.Score):  
        beatTime = beat_Times(score)
        df_Score = pd.DataFrame(index=beatTime)
        #df_Score['time'] = beatTime 
        #then run through all parts to first extract onsets and then build sequence
        k = 1
        for part in score:
            if isinstance(part,m21.stream.Part):
                if part.partName is None:
                    pn = 'Voice_' +str(k)
                    k+=1
                else:
                    pn = part.partName.strip('"')
                    if pn.startswith('Voice'):
                        pn = 'Voice_' +str(k)
                        k+=1
                onsets = Onset_Pitch_List(part)
                df_Score[pn] = Pitch_Beat_Interpolate(onsets[0],onsets[1],beatTime)
    return df_Score

def Comp_Decay(series,beat_decay_slope):
    # Assume evenly sampled index of series
    A = series.fillna(0)
    B = series
    step = np.min(np.diff(series.index))
    # ar values
    ar = -(beat_decay_slope*step)*np.arange(0,1./beat_decay_slope,step)
    # smoothing without negatives
    for i in range(len(ar)):
        shifted = B.shift(periods=i+1,fill_value=0) + ar[i]
        shifted = shifted.mask(shifted<0,0)
        A += shifted.fillna(0)
        #A = A.mask(A<0,0)
    return A

def Onset_Cues(score):
    beatTime = beat_Times(score)
    df_Score = pd.DataFrame(index=beatTime)
    k = 1
    for part in score:
        if isinstance(part,m21.stream.Part):
            pn = 'Voice_' + str(k)
            df_Score[pn] = Event_Beat_Interpolate(Onset_List(part),beatTime)
            k+=1
    df_seperate = df_Score.fillna(0)
    return df_seperate

def Motion_Cues(score):
    # cues to blend: co entry (++), co-onsets, co-direction (comodulation)
    # count up from bass, who shares with them?
    df_pitchs = Score_Pitch_Steps(score)
    beatwise = df_pitchs
    beatwise = beatwise.fillna(-25) # CHEAT to catch entries, assuming here no voice leaps more than two octaves
    beatwise = beatwise.diff()
    beatwise = beatwise.mask(beatwise>25,math.nan) # Mark entries with nan(cheat)
    beatwise = beatwise.mask(beatwise<-25,0) # ignore ends of lines 
    beatwise = beatwise.mask(beatwise>0,1) # equate all increases
    beatwise = beatwise.mask(beatwise<0,-1) # equate all decreases
    beatwise = beatwise.fillna(0) # remove entries by replacing nan with 0 
    beatwise = beatwise.mask(df_pitchs.isna(),math.nan) # reapply nan values where voices are quiet
    df_seperate = beatwise

    #returned columns of voice seperationg actions, with coaction attributed to highest active voice. Entries 1, other actions contribute 0.5
    # note: onsets counted only if with melodic motion.
    return df_seperate

def Entry_Cues(score):
    # cues to blend: co entry (++), co-onsets, co-direction (comodulation)
    # count up from bass, who shares with them?
    df_pitchs = Score_Pitch_Steps(score)
    beatwise = df_pitchs
    beatwise = beatwise.fillna(-25) # CHEAT to catch entries, assuming here no voice leaps more than two octaves
    beatwise = beatwise.diff()
    beatwise = beatwise.mask(beatwise>25,math.nan) # Mark entries with nan(cheat)
    beatwise = beatwise.mask(beatwise<20,0) # ignore ends of lines (cheat)
    beatwise = beatwise.fillna(2) # Distinguish entries by replacing nan with a higher values than changes in pitch
    beatwise = beatwise.mask(df_pitchs.isna(),math.nan) # reapply nan values where voices are quiet
    df_seperate = 0.5*beatwise
    #returned columns of voice seperationg actions, with coaction attributed to highest active voice. Entries 1, other actions contribute 0.5
    # note: onsets counted only if with melodic motion.
    return df_seperate

def Cue_Reduction(df_C):
    # for each voice, if voice seperation cues match those of a higher voice(s), make NA
    # or rather subtractive counting of independent streams, allowing for two or more voices per stream
    df_seperate = pd.DataFrame(index=df_C.index)
    parts = df_C.columns
    for i in range(len(parts)):
        df_seperate[parts[i]] = df_C[parts[i]]
    for i in range(len(parts)-1):
        for j in range(i+1,len(parts)):
            df_seperate[parts[j]] = df_seperate[parts[j]].mask(df_seperate[parts[j]]==df_seperate[parts[i]],math.nan)
    #df_seperate = 0.5*df_seperate.abs()
    df_seperate = df_seperate.fillna(0)
    return df_seperate

def Streaming(score):
    # TODO check if it is a score (m21 datatype) with multiple parts
    
    # calculate the streaming effect of this combination of cues 
    # Onsets
    df_C = Onset_Cues(score)
    df_seperate = Cue_Reduction(df_C)
    
    # add Pitch changes (with a reduction to remove reduncancy with onsets)
    df_C = Motion_Cues(score)
    df_seperate = df_seperate.mask(df_C.abs()>0,0)
    df_seperate = 0.25*df_seperate + 0.5*Cue_Reduction(df_C).abs()

    # Entries (with a reduction to removed redundancies with onsets)
    df_C = Entry_Cues(score)
    df_seperate = Cue_Reduction(df_C) + df_seperate # entries get twice the weight as other cues
    df_seperate = df_seperate.mask(df_seperate>1,1)

    # integrate successive cues and allow them decay time
    df_streams = Comp_Decay(df_seperate,0.2) # reduction linear over 4 steps
    df_streams = df_streams.mask(df_streams>1,1) # capt contributions per stream
    
    # find min streams WITH 1 held line if present
    
    df_sounds = Score_Pitch_Steps(score)       
    # B this series knows where the voices are silent. OK
    df_B = df_sounds.mask(df_sounds.isna(),0)
    B = df_B.sum(1)
    B =  B.mask(B==0,math.nan) 
    
    df_sounds = df_sounds.mask(df_sounds>1,0) # all times sounding is zero. 
    df_sounds += df_streams # add all the cues
    parts = df_sounds.columns 
    for i in range(len(parts)-1): # reduce the shared held parts to highest voice
        for j in range(i+1,len(parts)):
            df_sounds[parts[j]] = df_sounds[parts[j]].mask(df_sounds[parts[j]]==df_sounds[parts[i]],math.nan)
    # now all the zeros are the top held voice, but the cues are messed up
    df_sounds = df_sounds.mask(df_sounds>0,math.nan) # remove remaining cues
    df_sounds = df_sounds.mask(df_sounds==0,2) # isolate held lines above
    df_sounds = df_sounds.mask(df_sounds.isna(),0) # set
    df_sounds += df_streams # now the min voices are active, cues and held
    df_sounds = df_sounds.mask(df_sounds==0,math.nan) # remove the extraneous zeros
    df_streams = df_sounds.mask(df_sounds==2,0.25) # held notes are given some value but not a full stream

    C = df_streams.sum(1) - df_streams.max(1) + 1 # replace most active "distinct" voice with 1

    C = C.mask(C<1,1)
    C = C.mask(B.isna(),math.nan)
    df_streams['Total'] = C
    
    return df_streams

def Streaming_2(score):
    # TODO check if it is a score (m21 datatype) with multiple parts
    df_streams = All_Cues_Reduced(score)
    # find min streams WITH 1 held line if present
    
    df_sounds = Score_Pitch_Steps(score)       
    # B this series knows where the voices are silent. OK
    df_B = df_sounds.mask(df_sounds.isna(),0)
    B = df_B.sum(1)
    B =  B.mask(B==0,math.nan) 
    
    df_sounds = df_sounds.mask(df_sounds>1,0) # all times sounding is zero. 
    df_sounds += df_streams # add all the cues
    parts = df_sounds.columns 
    for i in range(len(parts)-1): # reduce the shared held parts to highest voice
        for j in range(i+1,len(parts)):
            df_sounds[parts[j]] = df_sounds[parts[j]].mask(df_sounds[parts[j]]==df_sounds[parts[i]],math.nan)
    # now all the zeros are the top held voice, but the cues are messed up
    df_sounds = df_sounds.mask(df_sounds>0,math.nan) # remove remaining cues
    df_sounds = df_sounds.mask(df_sounds==0,2) # isolate held lines above
    df_sounds = df_sounds.mask(df_sounds.isna(),0) # set
    df_sounds += df_streams # now the min voices are active, cues and held
    df_sounds = df_sounds.mask(df_sounds==0,math.nan) # remove the extraneous zeros
    df_streams = df_sounds.mask(df_sounds==2,0.25) # held notes are given some value but not a full stream

    C = df_streams.sum(1) - df_streams.max(1) + 1 # replace most active "distinct" voice with 1

    C = C.mask(C<1,1)
    C = C.mask(B.isna(),math.nan)
    df_streams['Total'] = C
    
    return df_streams

# now for onset only measure of seperation
def Event_Beat_Interpolate(events,beatTime):
    # fit onset list into beatTime sequence
    beatTime_es = np.array(np.zeros_like(beatTime), dtype=np.float)
    beatTime_es.fill(np.nan)
    for es in events:
        i = np.max(np.where(beatTime<=es))
        beatTime_es[i] = 1
    return beatTime_es

def Voice_Count(score):
    df_pitch = Score_Pitch_Steps(score)
    # get the number of voices active per moment 
    df_pitch = df_pitch.mask(df_pitch>1,1)
    df_pitch = Comp_Decay(df_pitch,0.2)
    df_pitch = df_pitch.mask(df_pitch>1,1)
    df_pitch = df_pitch.mask(df_pitch.isna(),0)
    df_active = df_pitch.sum(1)
    return df_active

def Distribution_Functions(A):
    # A is a single column pandas series
    # output pandas series with column 1: df, column 2: cdf
    # get cdf of total 
    Fcnt = Counter()
    for c in A:
        r=np.around(c,2)
        Fcnt[r]+=1
    dist_values = sorted(list(OrderedDict.fromkeys(np.around(A,2))))
    cdf_A = pd.DataFrame(index=dist_values)
    B = []
    C = []
    L = float(len(A))
    for i in range(len(dist_values)):
        B.append(Fcnt[dist_values[i]]/L)
        C.append(np.sum(B))
    cdf_A['df'] = B
    cdf_A['cdf'] = C
    return cdf_A

def All_Cues(score):
        # Onsets
    df_seperate  = Onset_Cues(score) 
    
    # Pitch changes (with a reduction to remove reduncancy with onsets)
    df_C = Motion_Cues(score).abs()
    df_seperate = df_seperate.mask(df_C.abs()>0,0)
    df_seperate = 0.25*df_seperate + 0.5*df_C.abs()
    
    # Entries (with a reduction to removed redundances with onsets)
    df_seperate += Entry_Cues(score) # entries get twice the weight as other cues
    df_seperate = df_seperate.mask(df_seperate>1,1)

    # integrate successive cues and allow them decay time
    df_streams = Comp_Decay(df_seperate,0.2) # reduction linear over 4 steps
    df_seperate = df_streams.mask(df_streams>1,1) # capt contributions per stream
    

    return df_seperate

def All_Cues_Reduced(score):
    # TODO check if it is a score (m21 datatype) with multiple parts
    
    # calculate the streaming effect of this combination of cues 
    # Onsets
    df_C = Onset_Cues(score)
    df_seperate = Cue_Reduction(df_C)
    
    # add Pitch changes (with a reduction to remove reduncancy with onsets)
    df_C = Motion_Cues(score)
    df_seperate = df_seperate.mask(df_C.abs()>0,0)
    df_seperate = 0.25*df_seperate + 0.5*Cue_Reduction(df_C).abs()

    # Entries (with a reduction to removed redundancies with onsets)
    df_C = Entry_Cues(score)
    df_seperate = Cue_Reduction(df_C) + df_seperate # entries get twice the weight as other cues
    df_seperate = df_seperate.mask(df_seperate>1,1)

    # integrate successive cues and allow them decay time
    df_streams = Comp_Decay(df_seperate,0.2) # reduction linear over 4 steps
    df_streams = df_streams.mask(df_streams>1,1) # capt contributions per stream
    
    return df_streams