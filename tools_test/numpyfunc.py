import numpy as np
import random
dimention=10
order_list=[[],[],[],[],[],[],[],[],[],[]]
x=np.random.random((50,dimention))
y=np.zeros((10000))
for i in (x):
    y=int(i[0]/0.1)
    order_list[y].append(i)

total_roads=100
path_length=10
total_label=5
final_x=np.zeros((total_roads,path_length,dimention))
final_y=np.zeros((total_roads,path_length))
for one_sample in range(total_roads):
    label_list=random.sample(range(dimention),total_label)
    one_shoot_label=label_list[-1]
    for i in range(path_length-2):
        label=np.random.choice(label_list[:-1])
        final_y[one_sample,i]=label
        final_x[one_sample,i,:]=np.random.choice(order_list[label])
    insert_idx=np.random.randint(0,path_length-2)