# Code for the Musicnn ConvNet linked to a lighting console.

from pynput.keyboard import Controller
import time
import sys
from musicnn.extractor import extractor
# %matplotlib inline
import numpy as np
from numpy.core.function_base import linspace
import matplotlib.pyplot as plt
keyboard = Controller()

# Initialization of the input and ouput files 
file_name = sys.argv[1]
print(file_name)
complete_file_name = file_name+'.wav'

# Call to the NN
taggram, tags = extractor(complete_file_name, model='MTT_musicnn', input_length=0.5, extract_features=False)
plt.rcParams["figure.figsize"] = (40,10) 
fontsize = 12 

# Plot the taggram
fig, ax = plt.subplots()
# title
ax.title.set_text('Taggram '+file_name)
ax.title.set_fontsize(fontsize)
# x-axis title
ax.set_xlabel('(seconds)', fontsize=fontsize)
# y-axis
y_pos = np.arange(len(tags))
ax.set_yticks(y_pos)
ax.set_yticklabels(tags, fontsize=fontsize-1)
# x-axis
step = 2
step_2 = 2
x_pos = np.arange(0, taggram.shape[0], step*step_2)
x_label = np.arange(0, taggram.shape[0]/step_2, step, dtype=int)%60
ax.set_xticks(x_pos)
ax.set_xticklabels(x_label, fontsize=fontsize-2)
# depict taggram
ax.imshow(taggram.T, interpolation=None, aspect="auto")
# plt.show()
fig.savefig('dmx_'+file_name, dpi=300)



########################################
# Detecting scene changes
last_action = '1'

def get_action(switch, last_action):
    a = ''
    while True:
        if   switch == 1:
            a = np.random.choice(['q', 'w', 'e'])
        elif switch == 2:
            a = np.random.choice(['r', 't', 'y'])
        elif switch == 3:
            a = np.random.choice(['a', 's', 'd'])
        elif switch == 4:
            a = np.random.choice(['f', 'g', 'h'])
        elif switch == 5:
            a = np.random.choice(['z', 'x', 'c'])
        elif switch == 6:
            a = np.random.choice(['v', 'b', 'n'])
        elif switch == 9:
            a = 'p'
        
        if last_action != a:
            break
        
    return a

times = len(taggram)
switches = np.zeros(times, dtype=int)

# Thresholds definitions
thres_total_energy = 2.3
thres_ok_top5 = 0.5
thres_drop_top5 = 0.02
threshold_voc = 0.15
thres_peak = 0.55

#Song sections detectors

# Calculating total energy per timestep.
totals = np.zeros(times)
for i in range(times):
  totals[i] = sum(taggram[i])
  if abs(totals[i]-totals[i-1]) > thres_total_energy:
    switches[i] = 2

# Finding TOP-10 tags per timestep.
top5 = np.zeros((times, 10), dtype=int)
for i in range(times):
  indices = np.argsort(-taggram[i])
  top5[i] = indices[:10]
  differences = np.in1d(top5[i],top5[i-1])
  if np.all(~differences): switches[i] = 3

  # One of the top5 with magnitude ok, drops
  for tag in top5[i-1]:
    if taggram[i-1][tag] > thres_ok_top5 and taggram[i][tag] < thres_drop_top5 : 
      switches[i] = 4
  
# Peak in a channel
  for t in range(len(tags)):
    if taggram[i][t] > thres_peak : switches[i] = 6

# Vocals / No Vocals
voc_tag, novoc_tag = 20, 21
voc_buffer = np.zeros((2, 2))
for i in range(times):
  v, nv = taggram[i][voc_tag], taggram[i][novoc_tag]
  avg_v, avg_nv = np.mean(voc_buffer[0]), np.mean(voc_buffer[1])
  if abs(v-avg_v)>threshold_voc or abs(nv-avg_nv)>threshold_voc:
    switches[i] = 5
  voc_buffer[0][i%2], voc_buffer[1][i%2] = v, nv

# Initial scene:
switches[0] = 1

# Printing the output scenes sequence
timelapse = linspace(0, 500, 1001)
print('Original output')
for j in range(times):
    if switches[j] != 0:
        print(int(np.floor(timelapse[j]/60)) ,np.around(timelapse[j]%60, 1), 'Switch:', switches[j])

# Keeping the same config for 10 timesteps at least  
for i in range(len(switches)):
  if switches[i] != 0:
    switches[i+1:i+18] = 0

# Last scene:
switches[len(switches)-1] = 9

# Printing the modified scenes sequence
print('############# Resulting output ################')
for j in range(times):
    if switches[j] != 0:
        print(int(np.floor(timelapse[j]/60)) ,np.around(timelapse[j]%60, 1), 'Switch:', switches[j])

print('#############################')


########################################
# DEMO
print(10)
time.sleep(5)
print(5)
time.sleep(1)
print(4)
time.sleep(1)
print(3)
time.sleep(1)
print(2)
time.sleep(1)
print(1)
time.sleep(1)
print('GO!')

t = time.localtime()
print('Starting song at', time.strftime("%H:%M:%S", t), 'which lasts', times/2, 'seconds')

for i in range(len(switches)):
  if switches[i] != 0:
    a = get_action(switches[i], last_action)
    last_action = a
    keyboard.press(str(a))
    keyboard.release(str(a))
  time.sleep(0.49)

print('END')
t = time.localtime()
print('Ending song at ', time.strftime("%H:%M:%S", t))


########################################
