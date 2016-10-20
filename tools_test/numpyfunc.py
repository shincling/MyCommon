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
    insert_idx=np.random.randint(0,path_length-1)
    final_x[one_sample,insert_idx]=random.sample(order_list[one_shoot_label],1)[0]
    final_y[one_sample,insert_idx]=one_shoot_label
    final_x[one_sample,-1]=random.sample(order_list[one_shoot_label],1)[0]
    final_y[one_sample,-1]=one_shoot_label
    for i in range(path_length-1):
        if i!=insert_idx:
            label=np.random.choice(label_list[:-1])
            final_y[one_sample,i]=label
            final_x[one_sample,i,:]=random.sample(order_list[label],1)[0]
