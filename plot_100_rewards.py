import matplotlib.pyplot as plt
import numpy as np

N = 10
step_scale = 500
swapping_step = 15000

steps = []
rewards = []
all_safe = []
stdevs =[]

# begin = 9
begin = 32
# min_line_length = 30

# begin = 52
min_line_length = 40

# starts_with = './models/'
starts_with = './experiments/'
# starts_with = 'D:/workspaces/work'

with open('output_with_std.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith(starts_with) and len(line) > min_line_length:
            # print(line)

            split = line.split(' ')

            # start_index = line.index(' ')
            # end_index = line.index(' ', start_index + 1)
            # reward = float(line[start_index:end_index])
            # step_num = int(line[begin:start_index])
            step_num = int(split[0][split[0].rindex('/')+1:])
            reward = split[1]

            # ddg_start = end_index
            # ddg_end = line.index(' ', ddg_start+1)
            # ddg_score = int(line[ddg_start:ddg_end])

            # sag_start = ddg_end
            # sag_end = line.index(' ', sag_start+1)
            # sag_score = int(line[sag_start:sag_end])

            # all_start = sag_end
            # all_end = line.index(' ', all_start+1)
            # all_score = int(line[all_start:all_end])

            stdev = float(split[-1][:-1])

            steps.append(step_num)
            rewards.append(reward)
            # all_safe.append(ddg_score)
            stdevs.append(stdev)
            # print(reward, line)

# rewards = [-14.1, 8.57, 253.1, 124.49, 162.08, 154.84, 383.7, 510.55, 726.04, 416.72, 144.1, 270.69, 650.78, 535.96, 496.87, 665.49, 167.37, 258.53, 447.64, 881.17, 893.19, 860.03, 782.09, 734.16, 769.81, 518.39, 815.49, 742.3, 830.49, 716.84, 1094.31, 811.94, 804.33, 1043.23, 1136.27, 834.94, 1270.72, 1122.82, 546.74, 1233.56, 1204.0, 908.04, 1275.24, 1187.77, 739.99, 984.3, 750.43, 791.02, 990.44, 873.01, 699.3, 1026.8, 1074.31, 595.46, 832.68, 1096.16, 1325.04, 1180.88, 1167.96, 753.23, 1408.33, 1401.41, 1231.01, 1271.43, 1233.0, 1336.37, 1180.24, 1348.75, 1537.78, 1326.55, 1347.55, 1452.57, 803.04, 1400.73, 881.15, 1092.39, 1097.78, 1399.05, 910.76, 1011.54, 1338.68, 1261.16, 1515.64, 1548.94, 1472.02, 1426.36, 1401.31, 1344.28, 1450.42, 1510.88, 1447.49, 1450.15, 1378.77, 1337.13, 1236.65, 1521.73, 1400.74, 1514.55, 1128.77, 1481.51, 1326.3, 1440.81, 1641.01, 1787.74, 1294.17, 1337.8, 1438.33, 1503.83, 1552.74, 1614.09, 1299.27, 1462.08, 1484.98, 1534.83, 1325.84, 1031.33, 1410.89, 1534.39, 1046.61, 1609.44]
# steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000, 45500, 46000, 46500, 47000, 47500, 48000, 48500, 49000, 49500, 50000, 50500, 51000, 51500, 52000, 52500, 53000, 53500, 54000, 54500, 55000, 55500, 56000, 56500, 57000, 57500, 58000, 58500, 59000, 59500, 60000]

rewards = [float(reward) for _, reward in sorted(zip(steps, rewards))]
all_safe = [safe for _, safe in sorted(zip(steps, all_safe))]
steps.sort()

print(f'rewards = {rewards}')
print(f'steps = {steps}')
print(f'all_safe = {all_safe}')
print('lengths', len(rewards), len(steps))

steps = np.array(steps)
rewards = np.array(rewards)
# all_safrewardse = np.array(all_safe)

rewards = np.array(rewards, dtype=float)
stdevs = np.array(stdevs, dtype=float)

moving_avg = np.cumsum(np.insert(rewards, 0, 0))
moving_avg = (moving_avg[N:] - moving_avg[:-N]) / float(N)

moving_avg_std = np.cumsum(np.insert(stdevs, 0, 0))
moving_avg_std = (moving_avg_std[N:] - moving_avg_std[:-N]) / float(N)

xs = [(x + N - 1) * step_scale for x in range(len(rewards) - N + 1)]

all_safe = np.array(all_safe)
all_safe_moving_avg = np.cumsum(np.insert(all_safe, 0, 0))
all_safe_moving_avg = (all_safe_moving_avg[N:] - all_safe_moving_avg[:-N]) / float(N)


# print(f'max score {max(rewards)}')
# print(f'max safe {max(all_safe)}')
# print(f'max score moving average {moving_avg.max()}')
# print(f'max all safe moving average {all_safe_moving_avg.max()}')
# print(f'number of perfect runs {np.count_nonzero(all_safe == 100)}')
# print(f'number of almost perfect runs {np.count_nonzero(all_safe == 99)}')
# print(f'number of 90-100 runs out of {len(all_safe)} models')
# for i in range(90, 101):
#     print(f'{i} -> {np.count_nonzero(all_safe == i)}  {100 * np.count_nonzero(all_safe == i) / len(all_safe) :.2}%')
#
# try:
#     print(f'max score {max(rewards)} at {steps[np.argmax(rewards)]} with {all_safe[np.argmax(rewards)]} safe and running average score {moving_avg[np.argmax(rewards)]}')
#     print(f'max safe {max(all_safe)} at {steps[np.argmax(all_safe)]} with a score of {rewards[np.argmax(all_safe)]} and running average score {moving_avg[np.argmax(all_safe)]}')
#     print(f'max score moving average {moving_avg.max()} at {steps[np.argmax(moving_avg)]} with safe of {all_safe[np.argmax(moving_avg)]}')
#     print(f'max all safe moving average {all_safe_moving_avg.max()} at {steps[np.argmax(all_safe_moving_avg)]}')
# except:
#     print('** error **')

# print(f'{np.argmax(rewards)} {np.argmax(moving_avg)}')

fig, ax1 = plt.subplots(figsize=(6.4*2, 4.8*2), dpi=300)

good_number = 1000
ax1.axhline(y=good_number, color='lightgreen', label='100% Goal')
ax1.axhline(y=good_number*0.9, color='gold', label='90% Goal')
ax1.axhline(y=good_number*0.85, color='orange', label='85% Goal')

# for i in range(len(rewards)):
#     if i * step_scale % swapping_step == 0 and i > 0:
#         ax1.axvline(x=i * step_scale, color='darkorchid', label='Brain Swaps')

upper = moving_avg + moving_avg_std
lower = moving_avg - moving_avg_std
ax1.fill_between(xs, upper, lower)

# ax1.plot(steps, rewards, label='Raw Rewards')
ax1.plot(xs, moving_avg, label=f'{N} Moving Avg', color='black')
plt.title('Rewards Over Time (100 Test)')
ax1.set_xlabel('Games')
ax1.set_ylabel('Rewards')
ax1.set_ylim([0, 1100])

# mid_steps = steps[(rewards >= good_number*0.85) & (rewards < good_number*0.9)]
# mid_rewards = rewards[(rewards >= good_number*0.85) & (rewards < good_number*0.9)]
# ax1.scatter(mid_steps, mid_rewards, marker='*', color='dimgray')

# upper_steps = steps[rewards >= good_number*0.9]
# upper_rewards = rewards[rewards >= good_number*0.9]
# ax1.scatter(upper_steps, upper_rewards, marker='*', color='black')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right')


# ax2 = ax1.twinx()
# ax2.set_ylabel('Wins')  # we already handled the x-label with ax1
# ax2.plot(steps, all_safe, color='black', label='DDG Safe')
# ax2.plot(xs, all_safe_moving_avg, color='green', label=f'{N} Moving Avg')
# ax2.tick_params(axis='y')
# ax2.set_ylim([0, 110])
# ax2.legend(loc='lower right')


plt.savefig("100_results.png")

plt.show()

# plt.clf()
# fig = plt.figure(figsize =(10, 7))
# plt.hist(all_safe, bins=100)
# plt.savefig("100_results_histogram.png")
# plt.show()