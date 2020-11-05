import pickle
import matplotlib.pyplot as plt
import plot_functions
import deal_with_files
import constants as c
NTag = 2

xmesh, ymesh, hmesh = deal_with_files.load_mesh(NTag, bins=(25,25,20))

print('files open')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolormesh(xmesh,ymesh,hmesh, shading='auto')
plot_functions.plotXhh()
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.title(f"Massplane, using data, NTag={NTag}")
plt.savefig(f"figures{c.bin_sizes}/fullmassplane_{NTag}tag_data.png")
plt.cla();plt.clf()
plt.close()

print('plotted')
