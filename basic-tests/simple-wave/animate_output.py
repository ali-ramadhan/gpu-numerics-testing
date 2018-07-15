import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-f', '--file', dest='filename', help='Data file to animate.')
args = parser.parse_args()

solution = np.loadtxt(args.filename, delimiter=',')

fig, ax = plt.subplots()

x = np.arange(0, 1, 0.01)
line, = ax.plot(x, solution[0, :])

plt.ylim(-2, 2)

def animate(i):
    line.set_ydata(solution[i, :])  # update the data
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 100), init_func=init, interval=25, blit=True)
plt.show()