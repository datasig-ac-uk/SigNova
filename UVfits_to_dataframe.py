import numpy as np
import os
from pyuvdata import UVData
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", '--filepath', type=str, help="Path to uvfits path")
parser.add_argument("-n", '--name', type=str, help="Pickle file name")
#opts.workDir = os.getenv("WorkDir")


opts = parser.parse_args() 

filepath = opts.filepath
#filepath = 'Simulation_CASA/Real_Data/QSO.J1048+4637.spw10_DiffTime.uvfits'  

####~~~~~~~~~~~~~~~~~Non uvfits data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
UV = UVData()
#UV.read_uvh5('Simulation_CASA/Real_Data/SimulClass_127Ant_50InTimes_384FrCh_ThermalNoise_SeedInitial_NoH1c.uvh5')
#UV.write_uvfits('Simulation_CASA/Real_Data/SimulClass_127Ant_50InTimes_384FrCh_ThermalNoise_SeedInitial_NoH1c.uvfits', force_phase=True, spoof_nonessential=True)
#
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UV.read(filepath)
#print(UV.baseline_array.shape)
#print(np.unique(UV.baseline_array))

data = []
for bl in np.unique(UV.baseline_array):
    data.append(UV.get_data(bl))
data=np.array(data)

print('Data', data.shape) 
print("UV Time: ",UV.time_array.shape)
print("UV Time: ",UV.Ntimes)
print("UV frequency: ",UV.Nfreqs)
print("UV pols: ",UV.Npols)

if data.ndim == 4:
    data = data[:,:,:,0] #Take only polarization 1
else:
    data = data

#time = data.shape[1]
#n_chann = data.shape[2]
time = UV.Ntimes
n_chann = UV.Nfreqs

print('time: {0}, channels: {1}'.format(time, n_chann))

#

##~~~~ZERO DATA WORK~~~~~~~~~~~~~~~
#mask = (data == 0)
#sec1 = np.where(mask.all(1))
#print('sec1', np.unique(sec1[0]))
#
#base_delete = np.unique(sec1[0])
#data = np.delete(data,base_delete,0)
#
#print('Data non zero', data.shape)
#for i in range(data.shape[0]):
#    for j in range(data.shape[2]):
#        if np.sum(data[i,:,j]) == 0:
#            print('ZERO HERE AGAIN: base{0}, FrCH{1} '.format(i,j))
#        else:
#            continue
#
######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#arr_ant_rem = data[:,2:52, :]
#print('Remove bad data shape', arr_ant_rem.shape)
#new = arr_ant_rem.reshape((arr_ant_rem.shape[0]*arr_ant_rem.shape[1]),arr_ant_rem.shape[2])

new = data
print('new', new.shape)

ants2 = UV.get_antpairs()

ants2 = np.array(ants2)

print('ORIGINAL ants', ants2.shape)

ants = []
for i in range(ants2.shape[0]):
    ants.append(np.repeat([ants2[i,:]], time,axis=0))
##
ants = np.array(ants)
print('MINE ANts Created', ants.shape)
#
##
if new.ndim > 1:
    ants = ants.reshape((ants.shape[0]*ants.shape[1]), ants.shape[2])
    cumsumnew = new.reshape((new.shape[0]*new.shape[1]),new.shape[2])
else:
    cumsumnew = new
    ants = ants
#
print('cumsum NEW', cumsumnew.shape)
print('DATA', new.shape)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Pandas~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

names = []
names.append('Ant1')
names.append('Ant2')

tog_p = np.column_stack((ants,cumsumnew))
#tog_p = np.column_stack((ants2,cumsumnew))
print('Ants cumsumnew', tog_p.shape)
print('DATA', new.shape)

for p in range(1,n_chann+1,1):
    names.append('FrCh{}'.format(p))
np_names = np.array(names)

df1 = pd.DataFrame(tog_p, columns=np_names)

final_data = []
for antenna_pair, df in df1.groupby(['Ant1', 'Ant2']):
    df = df.iloc[:,2:].apply(lambda x: pd.Series([x.values]), axis=0)
    df['Ant1'] = antenna_pair[0]
    df['Ant2'] = antenna_pair[1]
    final_data.append(df)
final_data = pd.concat(final_data)

final_data =  final_data.melt(id_vars=['Ant1', 'Ant2'], var_name="FrCh", value_name='Stream')
final_data['Stream'] = final_data['Stream'].apply(lambda x: np.hstack([np.real(x)[:,None], np.imag(x)[:,None]]))
print('Final data\n',final_data)
#

print('Shape', final_data.shape)
print('Compare', data[0,0,0])
print('Compare', data[1,0,0])
print('Compare', data[2,0,0])
print('Compare', data[3,0,0])
print('Compare', data[4,0,0])

##~~~~~~~~~~remove autocorrs for HERA

#final_data = final_data[final_data['Ant1'] != final_data['Ant2']]
#print('No AutoCorr\n',final_data)
#
print(len(final_data['Stream'].iloc[0]))
print(len(final_data['Stream'].iloc[10]))
print(len(final_data['Stream'].iloc[100]))
print(len(final_data['Stream'].iloc[200]))
print(len(final_data['Stream'].iloc[-1]))

#
final_data.to_pickle(opts.name+'.pkl')
