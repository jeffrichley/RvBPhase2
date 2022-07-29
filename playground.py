import numpy as np
from utils import minimax_value

u2 = np.random.rand(3, 3)
u3 = np.random.rand(3, 3)

print(u2)
print(u3)
print('************')

v2 = minimax_value(u2)
v3 = minimax_value(u3)
vall = minimax_value(u2 + u3)

print(v2, v3, v2 + v3)
print(vall)
print('************')

