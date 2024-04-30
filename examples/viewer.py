import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import pandas as pd


class AnimationData:

    def build_frame(self, keypoints):
        numpoints = len(keypoints[0])

        
        t = np.array([np.ones(numpoints) * i for i in range(len(keypoints))]).flatten()

        x = keypoints[:, :, 0].reshape([-1])
        y = keypoints[:, :, 1].reshape([-1])
        z = keypoints[:, :, 2].reshape([-1])

        df = pd.DataFrame({'time' : t,
                           'x' : x,
                           'y' : y,
                           'z' : z})
        
        return df

    def unpack_extras(self, data, used):
        # Clones are bones that always seem to have the same values as other bones
        clones = {
            31 : 30,
            28 : 27,
            24 : 13,
            16 : 13,
            23 : 22,
            20 : 19
        }

        # Fixed are bones that always seem to have the same value
        fixed = { 1 : np.array([-132.9486, 0, 0]),
                  6 : np.array([132.94882, 0, 0]),
                  11 : np.array([0, 0.1, 0])}
                  
        
        retval = np.zeros([data.shape[0], 32, 3])        
        for fromi, toi in enumerate(used):
            retval[:, toi, :] = data[:, fromi, :]

        for f in fixed:
            retval[:, f, :] = fixed[f]

        for c in clones:
            retval[:, c, :] = retval[:, clones[c], :]
            
        #np.savez("unpacked_data.npz", orig = data, unpacked = retval)
        return retval


    def build_lines(self, num):
        linex = []
        liney = []
        linez = []

        for f in self.used_bones:
            t = parents[f]
            if (t >= 0):
                linex.append([self.df.x[num * 32 + f], self.df.x[num * 32 + t]])
                liney.append([self.df.y[num * 32 + f], self.df.y[num * 32 + t]])
                linez.append([self.df.z[num * 32 + f], self.df.z[num * 32 + t]])

        return [linex, liney, linez]
    
    def __init__(self, data, extra_bones):

        self.used_bones = [2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]

        self.extra_bones = extra_bones

        if (not extra_bones):

            self.data = self.unpack_extras(data, self.used_bones)
        else:
            self.data = data

        self.df = self.build_frame(self.data)


class Animation:

    def drawlines(self, aidx, frame):
        linex, liney, linez = self.animdata[aidx].build_lines(frame)
        for idx in range(len(linex)):
            self.animlines[aidx].append(self.ax[aidx].plot(linex[idx], liney[idx], linez[idx]))

    def update_plot(self, frame):

        self.framecounter.set_text("frame=%d"%frame)

        for aidx, adata in enumerate(self.animdata):
            if (self.skellines):
                linex, liney, linez = adata.build_lines(frame)
                for idx in range(len(linex)):
                    self.animlines[aidx][idx][0].set_data_3d(linex[idx], liney[idx], linez[idx])

            if (self.dots):
                newdata = adata.df[adata.df['time'] == frame]
                self.animdots[aidx]._offsets3d = (newdata.x, newdata.y, newdata.z)

            
    def __init__(self, animations, dots = True, skellines = False, scale = 1.0, unused_bones = True):

        self.fig = plt.figure()
        self.skellines = skellines
        self.dots = dots
        self.scale = scale
        
        self.ax = []

        self.extra_bones = unused_bones

        self.frames = animations[0].shape[0]
        
        self.animdata = [AnimationData(anim, self.extra_bones) for anim in animations]

        self.animlines = []
        self.animdots = []
        
        for idx, adata in enumerate(self.animdata):
            self.ax.append(self.fig.add_subplot( 10 * len(animations) + 100 + (idx + 1), projection = '3d'))
            self.animlines.append([])
            idata = adata.df[adata.df['time'] == 0]

            if (self.skellines):
                self.drawlines(idx, 0)

            if (self.dots):
                self.animdots.append(self.ax[idx].scatter(idata.x, idata.y, idata.z))

            self.ax[idx].set_xlim(-self.scale, self.scale)
            self.ax[idx].set_ylim(-self.scale, self.scale)
            self.ax[idx].set_zlim(-self.scale, self.scale)

            self.ax[idx].view_init(elev = 90, azim = 270, roll = 0)

        
        self.framecounter = plt.figtext(0.1, 0.1, "frame=0")

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames = self.frames, interval = 16)
        plt.show()


kpf = np.load("keypoints.npz")
kp = kpf['keypoints']

#kp = kp.reshape([kp.shape[0], -1])

print(kp.shape)

anim = Animation([kp], dots = True, skellines = False, scale = 1000.0)
