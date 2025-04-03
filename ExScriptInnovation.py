#%%
import numpy as np
import matplotlib.pyplot as plt
from CalcInnovation import create_database, prediction_neighbors

#%% Example

N=2**16 #Size of realisations

#import Modane
path='/users2/local/e22froge/codes/Innovation/DataSignals/uspa_mob15_1.dat' #path to data
fid = open(path,'r')
signal = np.fromfile(fid, dtype='>f')
fid.close()
data=np.reshape(signal,(-1,N))


Nbibpow=16 # total number of point in which we search analogs


Bib=np.reshape(data,-1)
Bib=Bib[0:2**Nbibpow]
Bib=(Bib-np.mean(Bib))/np.std(Bib)
    
real=100 #Arbitrary chosen realisation on which to predict
velocity=(data[real,:]-np.mean(data[real,:]))/np.std(data[real,:])

p=3
db_analog =create_database(Bib,p=p+1,overlapping=False) #+1 for successor
db_pasts = create_database(velocity,p=p,overlapping=True)

vals= prediction_neighbors(db_pasts,db_analog, k=100)

innovation=velocity-vals
#Careful!! The p first values are Nan for alignment reasons          
plt.plot(innovation)

