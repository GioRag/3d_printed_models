# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:33:58 2017

@author: Giorgos
"""
import numpy as np
from sys import argv
from StringIO import StringIO
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.signal import savgol_filter, fftconvolve
import pickle

STATUS_OK = 0
STATUS_ERROR = -1


class DraggablePoint:

    # http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    lock = None #  only one can be animated at a time

    def __init__(self, parent, x=0.1, y=0.1, size=0.1):

        self.parent = parent
        self.point = patches.Ellipse((x, y), size, size * 3, fc='r', alpha=0.5, edgecolor='r')
        self.x = x
        self.y = y
        parent.figure.axes[0].add_patch(self.point)
        self.press = None
        self.background = None
        self.connect()

        if self.parent.list_points:
            line_x = [self.parent.list_points[0].x, self.x]
            line_y = [self.parent.list_points[0].y, self.y]

            self.line = Line2D(line_x, line_y, color='r', alpha=0.5)
            parent.fig.axes[0].add_line(self.line)


    def connect(self):

        'connect to all the events we need'

        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def on_press(self, event):

        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        if self == self.parent.list_points[1]:
            self.line.set_animated(True)
        else:
            self.parent.list_points[1].line.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_motion(self, event):

        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        if self == self.parent.list_points[1]:
            axes.draw_artist(self.line)
        else:
            self.parent.list_points[1].line.set_animated(True)
            axes.draw_artist(self.parent.list_points[1].line)

        self.x = self.point.center[0]
        self.y = self.point.center[1]

        if self == self.parent.list_points[1]:
            line_x = [self.parent.list_points[0].x, self.x]
            line_y = [self.parent.list_points[0].y, self.y]
            self.line.set_data(line_x, line_y)
        else:
            line_x = [self.x, self.parent.list_points[1].x]
            line_y = [self.y, self.parent.list_points[1].y]

            self.parent.list_points[1].line.set_data(line_x, line_y)

        # blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_release(self, event):

        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        if self == self.parent.list_points[1]:
            self.line.set_animated(False)
        else:
            self.parent.list_points[1].line.set_animated(False)

        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

        self.x = self.point.center[0]
        self.y = self.point.center[1]

    def disconnect(self):

        'disconnect all the stored connection ids'

        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


class LineBuilder(object):
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
#        x, y = zip(*self.poly.xy)
#        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
#        self.ax.add_line(self.line)
        self.x = []
        self.y = []
        self.fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.dblclick:
            if event.button == 1:
                if len(self.x) > 2:
                    self.__update_line()
                # Draw line
#                ax = plt.gca()
#                xy = plt.ginput(1)
                self.x.append(event.xdata)
                self.y.append(event.ydata)
                if len(self.x) == 2:
                    line = self.ax.plot(self.x, self.y, picker=5) # note that picker=5 means a click within 5 pixels will "pick" the Line2D object
                    E = (self.y[1] - self.y[0])/(self.x[1] - self.x[0])
                    print("E = %.2f" % E)
                    self.ax.figure.canvas.draw()  
#            elif event.button == 3:
#                # Write to figure
#                ax = plt.gca()
#                plt.figtext(3, 8, 'boxed italics text in data coords', style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
#                circ = plt.Circle((event.xdata, event.ydata), radius=0.07, color='g', picker = True)
#                ax.add_patch(circ)
#                ax.figure.canvas.draw()
            else:
                pass # Do nothing

    def __update_line(self):
        self.x = []
        self.y = []

def calculate_E(args):

    A = raw_input("Give the cross-section area of the sample (mm^2): ")
    L = raw_input("Give the reference lenght of the sample (mm): ")

    n = len(args)
    fig = plt.figure(figsize=(8, 10), facecolor='w', edgecolor='k',
                     linewidth=2.0, frameon=True)
    ax = fig.add_subplot(1, 1, 1)

    colours_ = ['b', 'r', 'g', 'm', 'c', 'k', 'y']
    _fileName = None
    ymax = []
    for i in range(1, n):
        _file = args[i]
        _fileName = _file.split("\\")

        data = np.genfromtxt(_file, delimiter=",", skip_header=4,
                             invalid_raise=False)
        t = data[:, 1]
        u = data[:, 2] - data[0, 2]
        f = data[:, 3] - data[0, 3]

        filtered1D = savgol_filter(f, 71, 3)
        filtered1D = filtered1D - filtered1D[0]

        sigma_e = f/float(A)
        ymax.append(np.max(sigma_e))
        sigma_e_f = filtered1D/float(A)
        epsilon_e = u/float(L)
        array2write = np.zeros((len(epsilon_e), 2))
        array2write[:, 0] = epsilon_e.copy()
        array2write[:, 1] = sigma_e_f.copy()
        pickle.dump(array2write, open("./" + _fileName[0] + "/" +
                                      _fileName[0] + "_%d.pkl" % i, 'wb'))
        ax.scatter(epsilon_e, sigma_e, alpha=0.25, c=colours_[i], marker='o',
                   s=15, label='unfiltered sample %d' % i)
        ax.plot(epsilon_e, sigma_e_f, colours_[i], label='sav_gol_%d' % i)
#    drs = []
#    circles = [patches.Circle((0.32, 0.3), 0.03, fc='r', alpha=0.5),
#               patches.Circle((0.3,0.3), 0.03, fc='g', alpha=0.5)]
#
#    for circ in circles:
#        ax.add_patch(circ)
#        dr = DraggablePoint(circ)
#        dr.connect()
#        drs.append(dr)
#    drawLine = LineBuilder(ax)
    ax.set_xlim(-0.01, 0.2)
    ax.set_ylim(-0.01, max(ymax))
    ax.set_xlabel("$\epsilon_e$")
    ax.set_ylabel("$\sigma_e$ (MPa)")
    plt.legend(loc="upper left", scatterpoints=1, ncol=2, fancybox=True,
               shadow=True)
    plt.title(_fileName[0])
#    plt.savefig(_fileName[0] + '.png')
    plt.show()

    return 0


def sgolay2d(z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1)*(order + 2)/2.0
    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = (band - np.abs(
                                           np.flipud(z[1:half_size+1, :]) -
                                           band))
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = (band + np.abs(np.flipud(
                                            z[-half_size-1:-1, :]) - band))
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = (band - np.abs(np.fliplr(
                                           z[:, 1:half_size+1]) - band))
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = (band + np.abs(np.fliplr(
                                            z[:, -half_size-1:-1]) - band))
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = (band - np.abs(np.flipud(np.fliplr(
                                 z[1:half_size+1, 1:half_size+1])) - band))
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = (band + np.abs(np.flipud(np.fliplr(
                                   z[-half_size-1:-1, -half_size-1:-1])) -
                                   band))

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = (band - np.abs(np.flipud(
                                  Z[half_size+1:2*half_size+1, -half_size:]) -
                                  band))
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = (band - np.abs(np.fliplr(Z[-half_size:,
                                  half_size+1:2*half_size+1]) - band))

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return (fftconvolve(Z, -r, mode='valid'),
                fftconvolve(Z, -c, mode='valid'))


def main(*args):
    calculate_E(argv)

    return STATUS_OK


if __name__ == "__main__":
    res = main()
    print(res)
