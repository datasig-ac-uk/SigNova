#~~~~~~~EXECUTE IT IN CASA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import pandas as pd
from numpy import random
from matplotlib.collections import LineCollection
from numpy import random
import re
from numpy import asarray
from numpy import savetxt


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", '--filepath', type=str, help="Path to .ms file")
parser.add_argument("-n", '--name', type=str, help="Pickle file name")

opts = parser.parse_args()

#msname = opts.filepath 

#msname = 'QSO.J1048+4637.spw10_noflags.ms'
msname = '1061318736.ms'
name = 'TRY'


tb.open( msname, nomodify=False )
dat = tb.getcol('DATA')
time = tb.getcol('TIME')
ant1 = tb.getcol('ANTENNA1')
ant2 = tb.getcol('ANTENNA2')
tdf = np.vstack((ant1,ant2,dat[0,:,:])).T #First porlarization
print('data', dat.shape)

n_chann = dat.shape[1]

unique_time = np.unique(time)
unique_ant = np.unique(ant1)
unique_ant1 = np.unique(ant1)
unique_ant2 = np.unique(ant2)
n_antennas = len(unique_ant)
print('Time:', len(unique_time))
print('Antennas1:', len(unique_ant1))
print('Antennas2:', len(unique_ant2))
#
print(tdf.shape)
###~~~~~~~~Check antenna/tile missing
#
#for i in range(n_antennas):
#    if i in ant1:
#        print('YES {}'.format(i))
#    else:
#        print('NO {}'.format(i))
#



###~~~ If you need to remove bad data like MWA:
ant = []

##For labels and complex shit
##for i in range(0,n_antennas+1):
for i,j in enumerate(unique_ant1):

    b = tdf[tdf[:, 0]== j,:]
#    for j in range(0, n_antennas+1):
#    for j in range( n_antennas):
    for k,l in enumerate(unique_ant2):
        c = b[b[:, 1]== l,:]
        if c.size:
            ant.append(c)
#            print('C!!', c)
        else:
            continue
##
arr_ant = np.array(ant)
print('arr_ant',arr_ant.shape)  #(7260, 56, 386)
#
arr_ant_rem = arr_ant[:,2:52, :]
new = arr_ant_rem.reshape((arr_ant_rem.shape[0]*arr_ant_rem.shape[1]),arr_ant_rem.shape[2])
print('New',new.shape)


##If not:
#new = tdf
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Pandas~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
names = []
names.append('Ant1')
names.append('Ant2')


for p in range(1,n_chann+1,1):
    names.append('FrCh{}'.format(p))
np_names = np.array(names)

df1 = pd.DataFrame(new, columns=np_names)

print('DataFrame\n', df1)

data = []
for antenna_pair, df in df1.groupby(['Ant1', 'Ant2']):
    df = df.iloc[:,2:].apply(lambda x: pd.Series([x.values]), axis=0)
    df['Ant1'] = antenna_pair[0]
    df['Ant2'] = antenna_pair[1]
    data.append(df)
data = pd.concat(data)
data =  data.melt(id_vars=['Ant1', 'Ant2'], var_name="FrCh", value_name='Stream')
data['Stream'] = data['Stream'].apply(lambda x: np.hstack([np.real(x)[:,None], np.imag(x)[:,None]]))
print(data)
#
print(len(data['Stream'].iloc[0]))
print(len(data['Stream'].iloc[10]))
print(len(data['Stream'].iloc[100]))
print(len(data['Stream'].iloc[200]))
print(len(data['Stream'].iloc[-1]))
#
#
#data.to_pickle(opts.name+'.pkl')
#data.to_pickle(name+'.pkl')

tb.close()
