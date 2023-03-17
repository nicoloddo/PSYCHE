#%%
'''
pandas: Geeft je pd.DataFrames laat maken die ongeveer werken als Excelsheets.
    Zij data een dataframe, dan kun je een kolom krijgen met data['kolom naam'].
    als je een deel van de kolom wil gebruik je data.loc[begin:eind, ['kolom naam']],
    je kan hier meerdere kolomnamen in invullen.
numpy: Gunt np.ndarray die eigenlijk lijsten zijn maar dan beter.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import csv



'''
datetime.datetime: Hiermee kan gemakkelijk van unix naar een datum worden gerekend
    en daarmee wat rekenen. Zie ook documentaties https://docs.python.org/3/library/datetime.html.
datetime.timedelta as time: Hiermee kan een relatieve tijd makkelijk worden bepaald en
    mee gerekend worden.
'''
from datetime import datetime
from datetime import timedelta, time 
'''
locale: geeft een hele hoop functies om verschillende komma conventies om te zetten.
    Hier hoef je dus niet echt iets mee te doen. Maar als je ast om een float heen
    zet, dan returnt deze de string met komma's voor decimalen etc. Met atof ga je
    van een string geschreven in de normale conventie (komma voor decimalen) naar
    een python float.
'''
from locale import atof, setlocale, LC_NUMERIC
from locale import str as ast
setlocale(LC_NUMERIC, 'nl_NL')

'''
De file paths naar de data en naar de timing dingen.
'''
data_path = 'resp_data/12051140_respdata.csv'
timing_path = 'interface_data/12051140_CNPthesisExperiment_2022_May_12_1149.csv'

with open(data_path) as file:
    file_data = pd.read_csv(file, delimiter = ';', skiprows = 1, dtype = str)[1:]
    file_data.set_index(file_data.iloc[:,0].values, inplace = True)
    
resp_data = file_data.iloc[:, [2, 3]]

start_respdata = time.fromisoformat(file_data.columns.values[13][10:-1])
start_respdata = (start_respdata.hour * 60 + start_respdata.minute) * 60 + start_respdata.second + start_respdata.microsecond*10**(-6)

resp_data = resp_data.fillna('-.0')
resp_data.columns = ['chest', 'abdomen']
resp_data = -resp_data.applymap(atof)
resp_data = resp_data.set_index((.01*int(i) for i in resp_data.index))
'''
data is een pd.DataFrame, voor de documentatie hierover zie
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html. Deze neemt de
data van data_path en leest deze in naar een pd.DataFrame. Deze neemt de kolommen
'chest' en 'abdomen' en maakt van alle lege cellen een 0 en zet alles om naar een float.
'''

baseline_length = 40
song_length = 40
speech_length = 30

with open(timing_path) as file:
    file_data = pd.read_csv(file, delimiter = ';', dtype = str)
    '''
    Check whether calibration time is in the columns, if it is use this to calculate the unix time.
    If not there should be a baseline unix.
    '''
    baseline_time = atof(file_data['baseline_time'][file_data['baseline_time'].first_valid_index()])
    baseline_onset = atof(file_data['eyesclosed2.started'][file_data['eyesclosed2.started'].first_valid_index()])
    speech_start = atof(file_data['text_16.started'][file_data['text_16.started'].first_valid_index()])
    
    songstart_rela = file_data.loc[file_data['song_start.started'].first_valid_index():file_data['song_start.started'].first_valid_index() + 4, 'song_start.started'].apply(atof).to_numpy()

#bereken het verschil tussen het begin van resp_data en het begin van het experiment in psypy.
#hele rare factor van 7200 wat 2 uur is maar weet ook niet helemaal waarom. Misschien zijn er ooit 2 uur geskipt ofzo.
time_delta = baseline_time%86400 - baseline_onset - start_respdata + 7200

baseline_data_abdomen, baseline_data_chest = resp_data.loc[10 + baseline_onset + time_delta:baseline_onset + baseline_length + time_delta].to_numpy().T
time_array = resp_data.loc[10 + baseline_onset + time_delta:baseline_onset + baseline_length + time_delta].index.to_numpy()
baseline_time_array = time_array - np.min(time_array)

#slice de song data voor chest en abdomen apart een lijstje van de vier songs
#werkt verder hetzelfde als baseline data

song_data_chest, song_data_abdomen, song_time_array = np.empty((3, 4, song_length*100 - 1000))
for i in range(4):
    chest, abdomen = resp_data.loc[10 + songstart_rela[i] + time_delta:songstart_rela[i] + song_length + time_delta].to_numpy().T
    song_data_chest[i] = chest
    song_data_abdomen[i] = abdomen
    time_array = resp_data.loc[10 + songstart_rela[i] + time_delta:songstart_rela[i] + song_length + time_delta].index.to_numpy()
    song_time_array[i] = time_array - np.min(time_array)

#slice speech data
speech_data_abdomen, speech_data_chest = resp_data.loc[speech_start + time_delta: speech_start + 30 + time_delta].to_numpy().T
time_array2 = resp_data.loc[speech_start + time_delta: speech_start + 30 + time_delta].index.to_numpy()
speech_time_array = time_array2 - np.min(time_array2)

speech_data_abdomen2, speech_data_chest2 = resp_data.loc[958.8333159 + time_delta: 958.8333159 + 30 + time_delta].to_numpy().T
time_array3 = resp_data.loc[958.8333159 + time_delta: 958.8333159 + 30 + time_delta].index.to_numpy()
speech_time_array2 = time_array3 - np.min(time_array3)

speech_data_abdomen3, speech_data_chest3 = resp_data.loc[720.563802 + time_delta: 720.563802 + 30 + time_delta].to_numpy().T
time_array4 = resp_data.loc[720.563802 + time_delta: 720.563802 + 30 + time_delta].index.to_numpy()
speech_time_array3 = time_array4 - np.min(time_array4)

speech_data_abdomen4, speech_data_chest4 = resp_data.loc[1175.161944 + time_delta: 1175.161944 + 30 + time_delta].to_numpy().T
time_array5 = resp_data.loc[1175.161944 + time_delta: 1175.161944 + 30 + time_delta].index.to_numpy()
speech_time_array4 = time_array5 - np.min(time_array5)


average_baseline_chest = np.average(baseline_data_chest)
average_baseline_abdomen= np.average(baseline_data_abdomen)
average_speech_chest1 = np.average(speech_data_chest)
average_speech_abdomen1 = np.average(speech_data_abdomen)
average_speech_chest2 = np.average(speech_data_chest2)
average_speech_abdomen2 = np.average(speech_data_abdomen2)
average_speech_chest3 = np.average(speech_data_chest3)
average_speech_abdomen3 = np.average(speech_data_abdomen3)
average_speech_chest4 = np.average(speech_data_chest4)
average_speech_abdomen4 = np.average(speech_data_abdomen4)


# speech_data_chest, speech_data_abdomen, speech_time_array = np.empty((3, 4, baseline_length*100 - 1000))
# for i in range(4):
#     chest, abdomen = resp_data.loc[speech_start[i]:speech_start[i] + baseline_length].to_numpy().T
#     speech_data_chest[i] = chest
#     speech_data_abdomen[i] = abdomen
#     time_array = resp_data.loc[speech_start[i]:speech_start[i] + 45].index.to_numpy()
#     speech_time_array[i] = time_array - np.min(time_array)
     

#filteren baseline
tick_spacing_x = 5 #verandert interval op axes
tick_spacing_y = 0.5 
sos = signal.butter(2, 1, 'low', fs=100, output='sos')
fil_baseline_chest = signal.sosfilt(sos, baseline_data_chest)
fil_baseline_abdomen = signal.sosfilt(sos, baseline_data_abdomen)

plt.figure('baseline', figsize = (12, 6))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7,4))
ax1.plot(baseline_time_array, fil_baseline_chest)
ax1.plot(baseline_time_array, fil_baseline_abdomen)
ax1.set_title('Low pass filtered (1Hz) chest respiration signal during baseline')
ax2.set_title('Low pass filtered (1Hz) abdomen respiration signal during baseline')
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax1.axhline(average_baseline_chest, color='grey', linestyle='dashed', linewidth = 0.8)
ax2.axhline(average_baseline_abdomen, color='grey', linestyle='dashed', linewidth = 0.8)
ax1.set_xlabel('Time [seconds]')
ax2.set_xlabel('Time [seconds]')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
plt.tight_layout()
plt.savefig("baselineplot.png", format="png", dpi=1200)
plt.show()


# speech part

sos3 = signal.butter(2, 1.5, 'low', fs=100, output='sos')
fil_speech_chest = signal.sosfilt(sos3, speech_data_chest)
fil_speech_abdomen = signal.sosfilt(sos3, speech_data_abdomen)
 

plt.figure('speech', figsize = (16, 10))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7,4))
plt.subplots_adjust(hspace = 0.8)
ax1.plot(speech_time_array, fil_speech_chest)
ax1.plot(speech_time_array, fil_speech_abdomen)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax1.set_title('Filtered chest respiration signal during speech after the first happy song')
ax2.set_title('Filtered abdomen respiration signal during speech after the first happy song')
ax1.set_xlabel('Time [seconds]')
ax2.set_xlabel('Time [seconds]')
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax1.axhline(average_speech_chest1, color='grey', linestyle='dashed', linewidth = 0.8)
ax2.axhline(average_speech_abdomen1, color='grey', linestyle='dashed', linewidth = 0.8)
plt.savefig('speechplot.png', format="png", dpi=1200)
plt.tight_layout()
plt.show()
 

 
sos4 = signal.butter(2, 1.5, 'low', fs=100, output='sos')
fil_speech_chest2 = signal.sosfilt(sos4, speech_data_chest2)
fil_speech_abdomen2 = signal.sosfilt(sos4, speech_data_abdomen2)

plt.figure('speech sad song', figsize = (16, 10))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7,4))
plt.subplots_adjust(hspace = 0.8)
ax1.plot(speech_time_array2, fil_speech_chest2)
ax1.plot(speech_time_array2, fil_speech_abdomen2)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax1.set_title('Filtered chest respiration signal during speech after the first sad song')
ax2.set_title('Filtered abdomen respiration signal during speech after the first sad song')
ax1.set_xlabel('Time [seconds]')
ax2.set_xlabel('Time [seconds]')
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax1.axhline(average_speech_chest2, color='grey', linestyle='dashed', linewidth = 0.8)
ax2.axhline(average_speech_abdomen2, color='grey', linestyle='dashed', linewidth = 0.8)
plt.savefig('speechplot2.png', format="png", dpi=1200)
plt.tight_layout()
plt.show()



 
sos4 = signal.butter(2, 1.5, 'low', fs=100, output='sos')
fil_speech_chest3 = signal.sosfilt(sos4, speech_data_chest3)
fil_speech_abdomen3 = signal.sosfilt(sos4, speech_data_abdomen3)

plt.figure('speech happy song', figsize = (16, 10))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7,4))
plt.subplots_adjust(hspace = 0.8)
ax1.plot(speech_time_array3, fil_speech_chest3)
ax1.plot(speech_time_array3, fil_speech_abdomen3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax1.set_title('Filtered chest respiration signal during speech after the second happy song')
ax2.set_title('Filtered abdomen respiration signal during speech after the second happy song')
ax1.set_xlabel('Time [seconds]')
ax2.set_xlabel('Time [seconds]')
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax1.axhline(average_speech_chest3, color='grey', linestyle='dashed', linewidth = 0.8)
ax2.axhline(average_speech_abdomen3, color='grey', linestyle='dashed', linewidth = 0.8)
plt.savefig('speechplot3.png', format="png", dpi=1200)
plt.tight_layout()
plt.show()


 
sos4 = signal.butter(2, 1.5, 'low', fs=100, output='sos')
fil_speech_chest4 = signal.sosfilt(sos4, speech_data_chest4)
fil_speech_abdomen4 = signal.sosfilt(sos4, speech_data_abdomen4)

plt.figure('speech sad song', figsize = (16, 10))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7,4))
plt.subplots_adjust(hspace = 0.8)
ax1.plot(speech_time_array4, fil_speech_chest4)
ax1.plot(speech_time_array4, fil_speech_abdomen4)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
ax1.set_title('Filtered chest respiration signal during speech after the second sad song')
ax2.set_title('Filtered abdomen respiration signal during speech after the second sad song')
ax1.set_xlabel('Time [seconds]')
ax2.set_xlabel('Time [seconds]')
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax1.axhline(average_speech_chest4, color='grey', linestyle='dashed', linewidth = 0.8)
ax2.axhline(average_speech_abdomen4, color='grey', linestyle='dashed', linewidth = 0.8)
plt.savefig('speechplot4.png', format="png", dpi=1200)
plt.tight_layout()
plt.show()


for i in range(4):
    #filteren
    average_song_chest = np.average(song_data_chest[i])
    average_song_abdomen = np.average(song_data_abdomen[i])
    sos2 = signal.butter(2, 1, 'low', fs=100, output='sos') 
    fil_song_chest = signal.sosfilt(sos2, song_data_chest[i])
    fil_song_abdomen = signal.sosfilt(sos2, song_data_abdomen[i])
    
    plt.figure(i, figsize = (12, 6))   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7, 4))
    plt.subplots_adjust(hspace = 0.8)
    ax1.plot(song_time_array[i], fil_song_chest)
    ax1.plot(song_time_array[i], fil_song_abdomen)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
    ax1.set_title(f'Filtered chest respiration signal during song {i + 1}')
    ax2.set_title(f'Filtered abdomen respiration signal during song {i + 1}')
    ax1.set_xlabel('Time [seconds]')
    ax2.set_xlabel('Time [seconds]')
    ax1.set_ylim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax1.axhline(average_song_chest, color='grey', linestyle='dashed', linewidth = 0.8)
    ax2.axhline(average_song_abdomen, color='grey', linestyle='dashed', linewidth = 0.8)
    plt.savefig(f'songsplot {i + 1}.png', format="png", dpi=1200)
    plt.tight_layout()
    plt.show()
    
    
# %%
