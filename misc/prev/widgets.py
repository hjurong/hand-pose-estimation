# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 19:27:22 2015

@author: JURONG
"""
# from __future__ import division, print_function
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button, RadioButtons

# ############################

# import numpy as np
# import matplotlib
# import matplotlib.colors as colors
# import matplotlib.patches as patches
# import matplotlib.mathtext as mathtext
# import matplotlib.pyplot as plt
# import matplotlib.artist as artist
# import matplotlib.image as image

# ####################################

# fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.25, bottom=0.25)
# t = np.arange(0.0, 1.0, 0.001)
# a0 = 5
# f0 = 3
# s = a0*np.sin(2*np.pi*f0*t)
# l, = plt.plot(t,s, lw=2, color='red')
# plt.axis([0, 1, -10, 10])

# axcolor = 'lightgoldenrodyellow'
# axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
# axamp  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

# sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
# samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

# def update(val):
#     amp = samp.val
#     freq = sfreq.val
#     l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#     fig.canvas.draw_idle()
# sfreq.on_changed(update)
# samp.on_changed(update)

# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# def reset(event):
#     sfreq.reset()
#     samp.reset()
# button.on_clicked(reset)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
# radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
# def colorfunc(label):
#     l.set_color(label)
#     fig.canvas.draw_idle()
# radio.on_clicked(colorfunc)

# plt.show()


# #####################################################
# #########################################################
# class ItemProperties:
#     def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
#                  alpha=1.0):
#         self.fontsize = fontsize
#         self.labelcolor = labelcolor
#         self.bgcolor = bgcolor
#         self.alpha = alpha

#         self.labelcolor_rgb = colors.colorConverter.to_rgb(labelcolor)
#         self.bgcolor_rgb = colors.colorConverter.to_rgb(bgcolor)

# class MenuItem(artist.Artist):
#     parser = mathtext.MathTextParser("Bitmap")
#     padx = 5
#     pady = 5
#     def __init__(self, fig, labelstr, props=None, hoverprops=None,
#                  on_select=None):
#         artist.Artist.__init__(self)

#         self.set_figure(fig)
#         self.labelstr = labelstr

#         if props is None:
#             props = ItemProperties()

#         if hoverprops is None:
#             hoverprops = ItemProperties()

#         self.props = props
#         self.hoverprops = hoverprops


#         self.on_select = on_select

#         x, self.depth = self.parser.to_mask(
#             labelstr, fontsize=props.fontsize, dpi=fig.dpi)

#         if props.fontsize!=hoverprops.fontsize:
#             raise NotImplementedError(
#                         'support for different font sizes not implemented')


#         self.labelwidth = x.shape[1]
#         self.labelheight = x.shape[0]

#         self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
#         self.labelArray[:, :, -1] = x/255.

#         self.label = image.FigureImage(fig, origin='upper')
#         self.label.set_array(self.labelArray)

#         # we'll update these later
#         self.rect = patches.Rectangle((0,0), 1,1)

#         self.set_hover_props(False)

#         fig.canvas.mpl_connect('button_release_event', self.check_select)

#     def check_select(self, event):
#         over, junk = self.rect.contains(event)
#         if not over:
#             return

#         if self.on_select is not None:
#             self.on_select(self)

#     def set_extent(self, x, y, w, h):
#         print(x, y, w, h)
#         self.rect.set_x(x)
#         self.rect.set_y(y)
#         self.rect.set_width(w)
#         self.rect.set_height(h)

#         self.label.ox = x+self.padx
#         self.label.oy = y-self.depth+self.pady/2.

#         self.rect._update_patch_transform()
#         self.hover = False

#     def draw(self, renderer):
#         self.rect.draw(renderer)
#         self.label.draw(renderer)

#     def set_hover_props(self, b):
#         if b:
#             props = self.hoverprops
#         else:
#             props = self.props

#         r, g, b = props.labelcolor_rgb
#         self.labelArray[:, :, 0] = r
#         self.labelArray[:, :, 1] = g
#         self.labelArray[:, :, 2] = b
#         self.label.set_array(self.labelArray)
#         self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

#     def set_hover(self, event):
#         'check the hover status of event and return true if status is changed'
#         b,junk = self.rect.contains(event)

#         changed = (b != self.hover)

#         if changed:
#             self.set_hover_props(b)


#         self.hover = b
#         return changed

# class Menu:

#     def __init__(self, fig, menuitems):
#         self.figure = fig
#         fig.suppressComposite = True

#         self.menuitems = menuitems
#         self.numitems = len(menuitems)

#         maxw = max([item.labelwidth for item in menuitems])
#         maxh = max([item.labelheight for item in menuitems])


#         totalh = self.numitems*maxh + (self.numitems+1)*2*MenuItem.pady


#         x0 = 100
#         y0 = 400

#         width = maxw + 2*MenuItem.padx
#         height = maxh+MenuItem.pady

#         for item in menuitems:
#             left = x0
#             bottom = y0-maxh-MenuItem.pady


#             item.set_extent(left, bottom, width, height)

#             fig.artists.append(item)
#             y0 -= maxh + MenuItem.pady


#         fig.canvas.mpl_connect('motion_notify_event', self.on_move)

#     def on_move(self, event):
#         draw = False
#         for item in self.menuitems:
#             draw = item.set_hover(event)
#             if draw:
#                 self.figure.canvas.draw()
#                 break


# fig = plt.figure()
# fig.subplots_adjust(left=0.3)
# props = ItemProperties(labelcolor='black', bgcolor='yellow',
#                        fontsize=15, alpha=0.2)
# hoverprops = ItemProperties(labelcolor='white', bgcolor='blue',
#                             fontsize=15, alpha=0.2)

# menuitems = []
# for label in ('open', 'close', 'save', 'save as', 'quit'):
#     def on_select(item):
#         print('you selected %s' % item.labelstr)
#     item = MenuItem(fig, label, props=props, hoverprops=hoverprops,
#                     on_select=on_select)
#     menuitems.append(item)

# menu = Menu(fig, menuitems)
# plt.show()



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

K = 1.0

def calculate_and_plot():
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(K*R)
    ax.cla()
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False, zorder=9001)

#Default plot
calculate_and_plot()

axcolor = 'lightgoldenrodyellow'
axs  = plt.axes([0.02,0.02, 0.7, 0.05], axisbg=axcolor)
samp = Slider(axs, 'K', 0.2, 5.0, valinit=1.0)

def update(k=0):
    global K
    K = samp.val
    calculate_and_plot()
    ax.plot([K],[K],[K], marker="o")
    plt.draw()
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    samp.reset()
button.on_clicked(reset)

plt.show()
