# -*- coding: utf-8 -*-
"""
Created on Sat May 09 19:33:20 2015

@author: JURONG
"""

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Button
#
#freqs = np.arange(2, 20, 3)
#
#fig, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)
#t = np.arange(0.0, 1.0, 0.001)
#s = np.sin(2*np.pi*freqs[0]*t)
#l, = plt.plot(t, s, lw=2)
#
#
#class Index:
#    ind = 0
#    def next(self, event):
#        self.ind += 1
#        i = self.ind % len(freqs)
#        ydata = np.sin(2*np.pi*freqs[i]*t)
#        l.set_ydata(ydata)
#        plt.draw()
#
#    def prev(self, event):
#        self.ind -= 1
#        i = self.ind % len(freqs)
#        ydata = np.sin(2*np.pi*freqs[i]*t)
#        l.set_ydata(ydata)
#        plt.draw()
#
#callback = Index()
#axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
#axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
#bnext = Button(axnext, 'Next')
#bnext.on_clicked(callback.next)
#bprev = Button(axprev, 'Previous')
#bprev.on_clicked(callback.prev)
#
#plt.show()



########################################################
########################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)
l, = plt.plot(t,s, lw=2, color='red')
plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()