import json
import matplotlib.pyplot as plt
import numpy as np

with open("./config.json", "rt") as f:
    config = json.load(fp=f)

def length(a, v):
    return (v * v * np.sin(2*a)) / 9.8

np.random.seed(config["seed"])

v_normal = np.random.normal(
        loc=config["velocity_normal"]["loc"],
        scale=config["velocity_normal"]["scale"],
        size=config["velocity_normal"]["size"]
        )
v_uniform = np.random.uniform(
        low=config["velocity_uniform"]["low"],
        high=config["velocity_uniform"]["high"],
        size=config["velocity_uniform"]["size"]
        )
a_uniform = np.random.uniform(
        low=np.deg2rad(config["angle_uniform"]["low"]),
        high=np.deg2rad(config["angle_uniform"]["high"]),
        size=config["angle_uniform"]["size"]
        )
a_normal = np.random.normal(
        loc=np.deg2rad(config["angle_normal"]["loc"]),
        scale=np.deg2rad(config["angle_normal"]["scale"]),
        size=config["angle_normal"]["size"]
        )

l_normal_normal = [length(a_normal, v_normal)]
l_normal_uniform = [length(a_normal, v_uniform)]
l_uniform_normal = [length(a_uniform, v_normal)]
l_uniform_uniform = [length(a_uniform, v_uniform)]


fig = plt.figure(figsize=(16, 10))
ax1 = plt.subplot(212)
ax1.margins(0.05)
ax1.hist(l_normal_normal, bins=50)
ax1.set_title('Length Normal/Normal')
ax1.set_xlim(0, 1100)
ax1.set_ylim(0, 40)

ax2 = plt.subplot(221)
ax2.margins(0.05)
ax2.hist(a_normal, bins=50)
ax2.set_title('Normal Angle')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 35)


ax3 = plt.subplot(222)
ax3.margins(0.05) 
ax3.hist(v_normal, bins=50)
ax3.set_title('Normal Velocity')
ax3.set_xlim(40, 110)
ax3.set_ylim(0, 35)


plt.savefig("LNN.png")

fig2 = plt.figure(figsize=(16, 10))
ax1 = plt.subplot(212)
ax1.margins(0.05)
ax1.hist(l_normal_uniform, bins=50)
ax1.set_title('Length Normal/Uniform')
ax1.set_xlim(0, 1100)
ax1.set_ylim(0, 50)

ax2 = plt.subplot(221)
ax2.margins(0.05)
ax2.hist(a_normal, bins=50)
ax2.set_title('Normal Angle')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 35)


ax3 = plt.subplot(222)
ax3.margins(0.05) 
ax3.hist(v_uniform, bins=50)
ax3.set_title('Uniform Velocity')
ax3.set_xlim(40, 110)
ax3.set_ylim(0, 35)


plt.savefig("LNU.png")

fig3 = plt.figure(figsize=(16, 10))
ax1 = plt.subplot(212)
ax1.margins(0.05)
ax1.hist(l_uniform_uniform, bins=50)
ax1.set_title('Length Uniform/Uniform')
ax1.set_xlim(0, 1100)
ax1.set_ylim(0, 40)

ax2 = plt.subplot(221)
ax2.margins(0.05)
ax2.hist(a_uniform, bins=50)
ax2.set_title('Uniform Angle')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 35)


ax3 = plt.subplot(222)
ax3.margins(0.05) 
ax3.hist(v_uniform, bins=50)
ax3.set_title('Uniform Velocity')
ax3.set_xlim(40, 110)
ax3.set_ylim(0, 35)


plt.savefig("LUU.png")

fig4 = plt.figure(figsize=(16, 10))
ax1 = plt.subplot(212)
ax1.margins(0.05)
ax1.hist(l_uniform_normal, bins=50)
ax1.set_title('Length Uniform/Normal')
ax1.set_xlim(0, 1100)
ax1.set_ylim(0, 40)

ax2 = plt.subplot(221)
ax2.margins(0.05)
ax2.hist(a_uniform, bins=50)
ax2.set_title('Uniform Angle')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 35)


ax3 = plt.subplot(222)
ax3.margins(0.05) 
ax3.hist(v_normal, bins=50)
ax3.set_title('Normal Velocity')
ax3.set_xlim(40, 110)
ax3.set_ylim(0, 35)


plt.savefig("LUN.png")
