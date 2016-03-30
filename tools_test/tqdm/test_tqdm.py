import time
from tqdm import tqdm

for i in tqdm(range(1000)):
    time.sleep(0.01)
    if not i%50:
        pass

with tqdm(total=100) as pbar:
    for i in range(10):
        pbar.update(10)