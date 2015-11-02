#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca> 
# File: visual.py

## Copyright (c) 2010-2015 Thomas Martin <thomas.martin.1@ulaval.ca>
## 
## This file is part of ORB
##
## ORB is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ORB is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.


import os
import logging

# VIEWER2D IMPORTS
from viewer2d.gtkw.ImageViewCanvasGtk import ImageViewCanvas
from viewer2d.gtkw.ImageViewCanvasTypesGtk import DrawingCanvas
import viewer2d.gtkw.Widgets
from viewer2d.gtkw import FileSelection, ColorBar
from viewer2d import AstroImage
from viewer2d import colors, cmap
from viewer2d.util import wcsmod
from viewer2d.util.wcsmod import AstropyWCS
wcsmod.use('astropy')

# ORB IMPORTS
from orb.core import Tools, Cube, HDFCube, Lines
import orb.utils

# OTHER IMPORTS
import gtk, gobject
import gtk.gdk
import numpy as np
import astropy.wcs as pywcs
import math

# MATPLOTLIB GTK BACKEND
from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import pylab as pl

gobject.threads_init()

import socket
from threading import Thread, Event

###########################################
### CLASS BASEVIEWER ######################
###########################################

class BaseViewer(object):
    """This class implements a basic viewer for FITS/HDF5
    cubes created with ORB modules."""

    filepath = None
    image = None

    init_set = None
    
    xy_start = None
    xy_stop = None
    
    mode = None
    key_pressed = None

    header = None
    dimx = None
    dimy = None
    dimz = None
    wcs = None

    hdf5 = None # True if the cube is an HDF5 Cube
    
    calib_map = None

    fitsimage = None
    root = None
    logger = None
    select = None
    canvas = None
    old_canvas_objs = None
    canvas_objs = None
    plot = None
    
    image_region = None
    
    autocut_methods = None

    zaxis = None
    wavenumber = None
    step = None
    order = None
    bunit = None

    spectrum_window = None

    DRAW_COLOR_DEFAULT = 'green'
    
    BAD_WCS_FLAG = False

    IMAGE_SIZE = 300

    MAX_CUBE_SIZE = 4. # Max cube size in Go
    MAX_CUBE_SECTION_SIZE = 1. # max section size to load in Go

    matplotlib_cmaps = ['gray', 'gist_rainbow', 'jet', 'cool', 'hot', 'autumn', 'summer', 'winter']
    
    def __init__(self, config_file_name='config.orb', no_log=True,
                 debug=False):

        """Init BaseViewer

        :param config_file_name: (Optional) Name of ORB config file
          (default 'config.orb').

        :param no_log: (Optional) If True no logfile will be created
          for ORB specific warnings, info and errors (default True).

        :param debug: (Optional) If True, all messages are printed on
          stdout (default False).
        """

        STD_FORMAT = '%(asctime)s | %(levelname)1.1s | %(filename)s:%(lineno)d (%(funcName)s) | %(message)s'
    
        logger = logging.getLogger("orb-viewer")
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(STD_FORMAT)

        if debug:
            stderrHdlr = logging.StreamHandler()
            stderrHdlr.setLevel(logging.INFO)
            stderrHdlr.setFormatter(fmt)
            logger.addHandler(stderrHdlr)
        
        self.logger = logger
        self.select = FileSelection.FileSelection()
        self.tools = Tools()
        self.tools._no_log = no_log
        self.tools.config_file_name = config_file_name
        self.init_set = True
        self.spectrum_channel = 1
        self.wcs_rotation = float(
            self.tools._get_config_parameter('WCS_ROTATION'))
        self.fov = float(
            self.tools._get_config_parameter('FIELD_OF_VIEW'))


        # crating colormaps
        self.colormaps = list()
        self.colormaps_name = list()
        for imap in self.matplotlib_cmaps:
           self.colormaps.append(cmap.matplotlib_to_ginga_cmap(pl.get_cmap(imap)))
           self.colormaps_name.append(imap)
           
        # Define GTK GUI
        root = gtk.Window(gtk.WINDOW_TOPLEVEL)
        root.set_title("ORB Viewer")
        root.set_border_width(2)
        root.connect("delete_event", lambda w, e: self._quit_cb(w))
        self.root = root
                         
        # define ImageView
        fi = ImageViewCanvas(logger)
        fake_image = AstroImage.AstroImage(logger=self.logger)
        fake_image.set_data(np.zeros((10,10), dtype=float))
        fi.set_image(fake_image)
        fi.enable_autocuts('on')
        fi.set_autocut_params('zscale')
        fi.set_color_map('gray')
        fi.enable_autozoom('on')
        fi.set_callback('none-move', self._mouse_motion_cb)
        fi.set_callback('button-press', self._start_box_cb)
        fi.set_callback('button-release', self._stop_box_cb)
        fi.set_callback('key-press', self._key_pressed_cb)
        fi.set_callback('key-release', self._key_released_cb)
        fi.set_bg(0.2, 0.2, 0.2)
        
        fi.ui_setActive(True)
        self.fitsimage = fi
        self.autocut_methods = self.fitsimage.get_autocut_methods()
        
        bd = fi.get_bindings()
        bd.enable_all(True)
    
        # canvas that we will draw on
        canvas = DrawingCanvas()
        canvas.enable_draw(True)
        canvas.set_drawtype(
            'rectangle', color=self.DRAW_COLOR_DEFAULT, alpha=1)
        canvas.setSurface(fi)
        self.canvas = canvas
        # add canvas to view
        fi.add(canvas)
        canvas.ui_setActive(True)

        w = fi.get_widget()
        w.set_size_request(self.IMAGE_SIZE, self. IMAGE_SIZE)

        # BIG BOX
        bigbox = gtk.VBox(spacing=2)

        # IMAGE FRAME
        imageframe = gtk.Frame('Image')
        imageframebox = gtk.VBox()

        # imageoptionsbox
        imageoptionsbox = gtk.HBox()
        
        # autocut
        autocutbox = gtk.VBox()
        wautocuts = viewer2d.gtkw.Widgets.ComboBox()
        for name in self.autocut_methods:
            wautocuts.append_text(name)
        wautocuts.set_index(4)
        self.autocut_index = 4
        wautocuts.add_callback('activated', self._set_autocut_method_cb)
        wautocuts_label = gtk.Label('Scale')
        autocutbox.pack_start(wautocuts_label)
        autocutbox.pack_start(wautocuts.widget)
        imageoptionsbox.pack_start(autocutbox, fill=False, expand=False)

        # colormap 
        colormapbox = gtk.VBox()
        wcolormap = viewer2d.gtkw.Widgets.ComboBox()
        for name in self.colormaps_name:
            wcolormap.append_text(name)
        wcolormap.set_index(0)
        wcolormap.add_callback('activated', self._set_colormap_cb)
        wcolormap_label = gtk.Label('Color Map')
        colormapbox.pack_start(wcolormap_label)
        colormapbox.pack_start(wcolormap.widget)
        imageoptionsbox.pack_start(colormapbox, fill=False, expand=False)

        # index scale
        indexbox = gtk.VBox()
        scalebox = gtk.HBox()
        self.wimage_index = gtk.Adjustment(value=0, lower=0,
                                           upper=100, step_incr=1,
                                           page_incr=10)
        
        self.wimage_index.connect('value-changed', self._set_image_index_cb)
        self.index_label = gtk.Label('Image index')
        index_scale = gtk.HScale(self.wimage_index)
        index_scale.set_digits(0)
        index_scale.set_value_pos(gtk.POS_RIGHT)
        index_scale.set_draw_value(False)
        indexbox.pack_start(self.index_label)
        scalebox.pack_start(index_scale)
        index_button = gtk.SpinButton(self.wimage_index)
        index_button.set_digits(0)
        scalebox.pack_start(index_button, expand=False)
        indexbox.pack_start(scalebox)
        imageoptionsbox.pack_start(indexbox, fill=True, expand=True)

        
        # space
        spacebox = gtk.HBox()
        imageoptionsbox.pack_start(spacebox, fill=True, expand=True)

        # save button
        savebox = gtk.VBox()
        saveimage_label = gtk.Label('')
        saveimage = gtk.Button('Save Image')
        saveimage.connect('clicked', self._save_image_cb)
        savebox.pack_start(saveimage)
        savebox.pack_start(saveimage_label)
        imageoptionsbox.pack_start(savebox, fill=False, expand=False)
        
        imageframebox.pack_start(imageoptionsbox, fill=False, expand=False)
        imageframebox.pack_start(w)
        imageframebox.pack_start(self._build_colorbar(), fill=True, expand=False)
        
        # Coordsbox
        coordsbox = gtk.HBox()
        ra_label = gtk.Label('RA ')
        self.ra = gtk.Label('')
        self.ra.set_width_chars(13)
        dec_label = gtk.Label(' DEC ')
        self.dec = gtk.Label('')
        self.dec.set_width_chars(13)
        x_label = gtk.Label(' X ')
        self.x = gtk.Label('')
        self.x.set_width_chars(10)
        y_label = gtk.Label(' Y ')
        self.y = gtk.Label('')
        self.y.set_width_chars(10)
        value_label = gtk.Label(' Value ')
        self.value = gtk.Label('')
        self.value.set_width_chars(10)
        for box in (ra_label, self.ra, gtk.VSeparator(),
                    dec_label, self.dec, gtk.VSeparator(),
                    x_label, self.x, gtk.VSeparator(),
                    y_label, self.y, gtk.VSeparator(),
                    value_label, self.value):
            coordsbox.pack_start(box, fill=False, expand=False)
        imageframebox.pack_start(coordsbox, fill=False, expand=False)

        # Statbox
        STAT_WIDTH = 10
        statsbox = gtk.HBox()
        mean_label = gtk.Label('MEAN ')
        self.mean = gtk.Label('')
        self.mean.set_width_chars(STAT_WIDTH)
        median_label = gtk.Label(' MED ')
        self.median = gtk.Label('')
        self.median.set_width_chars(STAT_WIDTH)
        std_label = gtk.Label(' STD ')
        self.std = gtk.Label('')
        self.std.set_width_chars(STAT_WIDTH)
        sum_label = gtk.Label(' SUM ')
        self.sum = gtk.Label('')
        self.sum.set_width_chars(STAT_WIDTH)
        surf_label = gtk.Label(' SURF ')
        self.surf = gtk.Label('')
        self.surf.set_width_chars(STAT_WIDTH)
        
        for box in (mean_label, self.mean, gtk.VSeparator(),
                    median_label, self.median, gtk.VSeparator(),
                    std_label, self.std, gtk.VSeparator(),
                    sum_label, self.sum, gtk.VSeparator(),
                    surf_label,
                    self.surf):
            statsbox.pack_start(box, fill=False, expand=False)

        imageframebox.pack_start(gtk.HSeparator(), fill=False, expand=False)
        imageframebox.pack_start(statsbox, fill=False, expand=False)

        imageframe.add(imageframebox)
        
        # IMAGE BOX
        imagebox = gtk.VBox(spacing=2)
        imagebox.pack_start(imageframe, fill=True, expand=True)
        
        
        # MENU
        def new_submenu(name, cb):
            _m = gtk.MenuItem(name)
            _m.connect('activate', cb)
            return _m
   
        file_menu = gtk.Menu()
        file_mi = gtk.MenuItem('File')
        file_mi.set_submenu(file_menu)
        
        submenus = list()
        submenus.append(new_submenu('Open...',
                                    self._open_file_cb))
        submenus.append(new_submenu('Display header...',
                                    self._display_header_cb))
        for submenu in submenus:
            file_menu.append(submenu)
        
        
        menu_bar = gtk.MenuBar()
        menu_bar.append(file_mi)

        self.regions = Regions(change_region_properties_cb=self._change_region_properties_cb)
        menu_bar.append(self.regions.get_menubar_item())
        
        # PACK BIGBOX
        bigbox.pack_start(menu_bar, fill=False, expand=False, padding=2)
        
        bigbox.pack_start(imagebox, fill=True, expand=True)

        # add plugins
        plugins = self._get_plugins()
        for plugin in plugins:
            bigbox.pack_start(plugin, fill=False, expand=False, padding=2)
        
        root.add(bigbox)

    def _build_colorbar(self):
        """Return a colorbar frame.

        Mostly copied from GingaGtk.build_colorbar()
        """
        # colorbar
        rgbmap = self.fitsimage.get_rgbmap()
        
        rgbmap.add_callback('changed', self._rgbmap_cb)
        cbar = ColorBar.ColorBar(
            self.logger, rgbmap=rgbmap, link=True)
        cbar.set_range(*self.fitsimage.get_cut_levels())
        cbar.show()
        fr = gtk.Frame()
        fr.set_shadow_type(gtk.SHADOW_ETCHED_OUT)
        fr.add(cbar)
        self.colorbar = cbar
        return fr

    def _get_plugins(self):
        """A list of plugins (returned as a list of gtk.Box instance)
        may be defined and incerted here. It will be packed vartically
        at the bottom of the basic view.
        """
        return []

    def _reload_file(self):
        """Reload a file and update the displayed data. The file might
        have a new size.
        """
        print ' > Reloading {}'.format(self.filepath)
        self.load_file(self.filepath, reload=True)


    def _change_region_properties_cb(self, regions):
        self.canvas.set_drawtype(
            regions.get_shape(), color=self.DRAW_COLOR_DEFAULT, alpha=1)

    def _rgbmap_cb(self, rgbmap):
        self.colorbar.set_range(*self.fitsimage.get_cut_levels())
        self.colorbar.redraw()

    def _set_colormap_cb(self, c, idx):
        self.fitsimage.set_cmap(self.colormaps[idx])
        
    def _open_file_cb(self, c):
        """open-file callback

        Popup a File chooser dialog to open a new file
        """
        self.init_set = True
        self.pop_file_chooser_dialog(self.load_file, mode='open')

    def _display_header_cb(self, c):
        """display-header callback

        Popup a window with the printed header
        """
        w = HeaderWindow(self.header)
        w.show()
        

    def _key_pressed_cb(self, c, key):
        """key-pressed callback.
        
        :param c: Caller Instance
        :param key: key pressed
        """
        
        if key == 'control_l':
            self.mode = 'ctrl'
        if key == 'shift_l':
            self.mode = 'shift'
        if key == 'alt_l':
            self.mode = 'alt'

        if self.mode in ['shift', 'ctrl', 'alt']:
            self.canvas.enable_draw(False)
        else:
            self.canvas.enable_draw(True)
            
        self.key_pressed = key

    def _key_released_cb(self, c, key):
        """key-released callback.
        
        :param c: Caller Instance
        :param key: key pressed
        """
        if key in ['control_l', 'shift_l', 'alt_l']:
            self.mode = None

        if self.mode in ['shift', 'ctrl', 'alt']:
            self.canvas.enable_draw(False)
        else:
            self.canvas.enable_draw(True)
            
        self.key_pressed = None

    def _set_image_index_cb(self, c):
        """value-changed callback

        Changed the index of the displayed image

        :param c: Caller Instance
        """
        val = int(c.get_value())
        
        if self.zaxis is not None:
            zval = self.zaxis[c.get_value()]
            if self.wavenumber:
                unit = 'cm-1'
                conv = '{:.2f} nm'.format(orb.utils.cm12nm(zval))
            else:
                unit = 'nm'
                conv = '{:.2f} cm-1'.format(orb.utils.nm2cm1(zval))
            self.index_label.set_text('{:.2f} {} ({})'.format(
                zval, unit, conv))
            
        self.update_image(self.cube[:,:,val])
    
    def _save_image_cb(self, c):
        """save-image callback

        Pop a file chooser dialog to save the displayed Image

        :param c: Caller Instance
        """
        self.pop_file_chooser_dialog(self.save_image, mode='save')


    def _set_autocut_method_cb(self, c, idx):
        """autocut-method-changed callback

        Changes the autocut method
        
        :param c: Caller Instance

        :param idx: Index of the new autocut method in
          self.autocut_methods list
        """
        self.autocut_index = idx
        self.fitsimage.set_autocut_params(self.autocut_methods[idx])
        self._rgbmap_cb(self.fitsimage.get_rgbmap())

    def _start_box_cb(self, c, button, data_x, data_y):
        """start-drawing-box callback

        Start drawing a box, delete old drawn boxes

        :param c: Caller instance

        :param button: button pressed while drawing (e.g. ctrl or shift)

        :param data_x: X position when mouse button pressed

        :param data_y: Y position when mouse button pressed
        """
        if self.mode not in ['ctrl', 'shift', 'alt'] and button == 4:
            if self.key_pressed == 'k':
                self.sky_mode = True
            else:
                self.sky_mode = False
                
            self.xy_start = (data_x, data_y)
            
            self.canvas.deleteObject(
                self.canvas_objs)
        
    def _stop_box_cb(self, c, button, data_x, data_y):
        """stop-drawing-box callback

        Stop drawing a box

        :param c: Caller instance

        :param button: button pressed while drawing (e.g. ctrl or shift)

        :param data_x: X position when mouse button released

        :param data_y: Y position when mouse button released
        """
        if self.mode not in ['ctrl', 'shift', 'alt'] and button == 4:    
            self.xy_stop = (data_x + 1, data_y + 1)

            self.image_region = None
            
            if self.regions.get_shape() == 'circle':
                center = np.array(self.xy_start)
                radius = math.sqrt(np.sum((np.array(self.xy_stop) - center)**2.))
                Y, X = np.mgrid[0:self.dimx, 0:self.dimy]
                R = np.sqrt((X - center[0])**2. + (Y - center[1])**2.)
                mask = (R <= radius)
                mask_pixels = np.nonzero(mask)
                self.image_region = self.image[mask_pixels]
                x_range = np.array([np.nanmin(mask_pixels[0]),
                                    np.nanmax(mask_pixels[0])])
                y_range = np.array([np.nanmin(mask_pixels[1]),
                                    np.nanmax(mask_pixels[1])])

            else:
                mask = None
                y_range = np.array([min(self.xy_start[0], self.xy_stop[0]),
                                    max(self.xy_start[0], self.xy_stop[0])])
                x_range = np.array([min(self.xy_start[1], self.xy_stop[1]),
                                    max(self.xy_start[1], self.xy_stop[1])])


            # check range
            x_range[np.nonzero(x_range < 0)] = 0
            x_range[np.nonzero(x_range > self.dimx)] = self.dimx
            y_range[np.nonzero(y_range < 0)] = 0
            y_range[np.nonzero(y_range > self.dimy)] = self.dimy

            if x_range[1] - x_range[0] < 1.:
                if x_range[0] < self.dimx:
                    x_range[1] = x_range[0] + 1
                else:
                    x_range = [self.dimx - 1, self.dimx]

            if y_range[1] - y_range[0] < 1.:
                if y_range[0] < self.dimy:
                    y_range[1] = y_range[0] + 1
                else:
                    y_range = [self.dimy - 1, self.dimy]

            if self.regions.get_shape() == 'rectangle':
                if self.image is not None:
                    self.image_region = self.image[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1]]

                
            if self.image_region is not None:
                self.mean.set_text('{:.3e}'.format(
                    np.nanmean(self.image_region)))
                self.median.set_text('{:.3e}'.format(
                    orb.utils.robust_median(self.image_region)))
                self.std.set_text('{:.3e}'.format(
                    np.nanstd(self.image_region)))
                self.sum.set_text('{:.3e}'.format(
                    np.nansum(self.image_region)))
                self.surf.set_text('{:.2e}'.format(
                    self.image_region.size))

            # register new canvas object
            new_obj = self._get_new_object()
            if new_obj is not None:
                self.canvas_objs = new_obj

            # plot spectrum
            if self.dimz > 1:
                if self.spectrum_window is None:
                    self.spectrum_window = ZPlotWindow(
                        self.step, self.order,
                        self.wavenumber, self.bunit)
                    self.spectrum_window.show()
                elif not self.spectrum_window.w.get_property('visible'):
                    self.spectrum_window = ZPlotWindow(
                        self.step, self.order,
                        self.wavenumber, self.bunit)
                    self.spectrum_window.show()

                if self.hdf5:
                    self.cube._silent_load = True

                zdata = self.cube[int(x_range[0]):int(x_range[1]),
                                  int(y_range[0]):int(y_range[1]), :]

                if self.regions.get_shape() == 'circle':
                    for ii in range(zdata.shape[0]):
                        for ij in range(zdata.shape[1]):
                            if not mask[ii+int(x_range[0]),
                                        ij+int(y_range[0])]:
                                zdata[ii,ij,:].fill(np.nan)
                                

                if len(zdata.shape) == 3:
                    if self.regions.get_method() == 'sum':
                        zdata = np.nansum(np.nansum(zdata, axis=0), axis=0)
                    elif self.regions.get_method() == 'mean':
                        zdata = np.nanmean(np.nanmean(zdata, axis=0), axis=0)
                    elif self.regions.get_method() == 'median':
                        zdata = np.nanmedian(np.nanmedian(zdata, axis=0), axis=0)
                    else: raise Exception('Method error')
                        
                        
                self.spectrum_window.update(zdata)
        
    def _mouse_motion_cb(self, c, button, data_x, data_y):
        """mouse-motion callback

        Called when the mouse is moving on the image display.
        
        Display some informations on the mouse position.

        :param c: Caller instance
        
        :param button: button pressed while moving (e.g. ctrl or shift)

        :param data_x: X position of the mouse on the image display.

        :param data_y: Y position of the mouse on the image display.
        """
        # Get the value under the data coordinates
        try:
            value = c.get_data(int(data_x+0.5), int(data_y+0.5))

        except Exception:
            value = None

        fits_x, fits_y = data_x + 1, data_y + 1

        # Calculate WCS RA/DEC
        try:
            ra, dec = self.wcs.wcs.wcs_pix2world(fits_y, fits_x, 0)
            ra_txt = '{:.0f}:{:.0f}:{:.1f}'.format(*orb.utils.deg2ra(ra))
            dec_txt = '{:.0f}:{:.0f}:{:.1f}'.format(*orb.utils.deg2dec(dec))
            
        except Exception as e:
            self.logger.warn("Bad coordinate conversion: %s" % (
                str(e)))
            ra_txt  = 'BAD WCS'
            dec_txt = 'BAD WCS'

        # hack: x/y inverted
        self.ra.set_text('{}'.format(ra_txt))
        self.dec.set_text('{}'.format(dec_txt))
        self.x.set_text('{:.2f}'.format(fits_y))
        self.y.set_text('{:.2f}'.format(fits_x))
        if value is not None:
            self.value.set_text('{:.2e}'.format(value))
        else:
            self.value.set_text('NaN')
        
    def _quit_cb(self, c):
        """quit callback.

        Quit viewer.

        :param c: Caller instance
        """
        gtk.main_quit()
        return True

    def _get_new_object(self):
        """Return new boxes on the image display."""
        objs = self.canvas.getObjects()
        if self.old_canvas_objs is None:
            self.old_canvas_objs = list(objs)
            return objs[0]
        else:
            for obj in objs:
                if obj not in self.old_canvas_objs:
                    self.old_canvas_objs = list(objs)
                    return obj
            self.old_canvas_objs = list(objs)
            return None
            
    def get_widget(self):
        """Return root widget"""
        return self.root
        
    def load_file(self, filepath, reload=False, chip_index=1):
        """Load the file to display. Can be a FITS or HDF5 cube.

        :param filepath: Path to the file.

        :param reload: (Optional) Must be set to True if the file is
          reloaded (default False)
        """
        if os.path.splitext(filepath)[-1] in ['.fits']:
            self.hdf5 = False
            hdu = self.tools.read_fits(filepath, return_hdu_only=True)
            if 'NAXIS3' in hdu[0].header:
                cube_size = float(hdu[0].header['NAXIS1']) * float(hdu[0].header['NAXIS2']) * float(hdu[0].header['NAXIS3']) * 4 / 1e9
            
                if cube_size > self.MAX_CUBE_SIZE:
                    raise Exception('Cube size is too large: {} Go > {} Go'.format(cube_size, self.MAX_CUBE_SIZE))

            # detect a sitelle file
            image_mode = 'classic'
            
            if 'DETECTOR' in hdu[0].header:
                if 'SITELLE' in hdu[0].header['DETECTOR']:
                    if float(hdu[0].header['NAXIS2']) > 4000:
                        image_mode = 'sitelle'
            if 'INSTRUME' in hdu[0].header:
                if 'SITELLE' in hdu[0].header['INSTRUME']:
                    if float(hdu[0].header['NAXIS2']) > 4000:
                        image_mode = 'sitelle'

            self.cube, self.header = self.tools.read_fits(
                filepath,
                return_header=True,
                image_mode=image_mode,
                chip_index=chip_index,
                memmap=False,
                dtype=np.float32)
            
            self.filepath = filepath

        elif os.path.splitext(filepath)[-1] in ['.hdf5']:
            self.hdf5 = True
            self.cube = HDFCube(filepath, no_log=True)
            self.header = self.cube.get_cube_header()
            self.filepath = filepath

        else:
            raise Exception('File must be a FITS of HDF5 cube')

        if len(self.cube.shape) == 2: self.cube = self.cube[:,:,np.newaxis]
        
        self.dimx, self.dimy, self.dimz = self.cube.shape

        
        # SET WCS
        self.wcs = AstropyWCS(self.logger)
        wcs_header = self.header               
        self.wcs.wcs = pywcs.WCS(wcs_header, naxis=2)
        
        try:
            ra, dec = self.wcs.wcs.wcs_pix2world(-1, -1, 0)
            if ra < 1e-15 and dec < 1e-15:
                raise Exception('BAD WCS')
            self.BAD_WCS_FLAG = False
        except Exception as e:
            print 'WCS Error: ', e
            self.BAD_WCS_FLAG = True

        if not reload:
            val = int(self.wimage_index.get_value())
            if val < 0 or val >= self.cube.shape[2]:
                val = 0
            image_data = self.cube[:,:,val]    
            self.update_image(image_data)
            

        # SET ZAXIS
        self.wimage_index.set_upper(self.dimz-1)
        
        if self.header is not None:
            if 'ORDER' in self.header:
                self.order = self.header['ORDER']
            elif 'SITORDER' in self.header:
                self.order = self.header['SITORDER']
            if 'STEP' in self.header:
                self.step = self.header['STEP']
            elif 'SITSTPSZ' in self.header:
                self.step = self.header['SITSTPSZ'] * self.header['SITFRGNM']
                
            if 'CUNIT3' in self.header:
                if 'nm' in self.header['CUNIT3']:
                    self.wavenumber = False
                elif 'cm-1' in self.header['CUNIT3']:
                    self.wavenumber = True
                
            if (self.order is not None
                and self.step is not None
                and self.wavenumber is not None):
                if self.wavenumber:
                    self.zaxis = orb.utils.create_cm1_axis(
                        self.dimz, self.step, self.order)
                else:
                    self.zaxis = orb.utils.create_nm_axis(
                        self.dimz, self.step, self.order)

            if 'BUNIT' in self.header:
                if 'erg/cm^2/s/A' in self.header.comments['BUNIT']:
                    self.bunit = 'erg/cm^2/s/A'
                else:
                    self.bunit = self.header.comments['BUNIT']
                    
        self.root.set_title(filepath)


    def pop_file_chooser_dialog(self, action, mode='save'):
        """Pop a a file chooser dialog

        :param action: Method launched when file path has been
          defined.
        """
        if mode == 'save':
            act = gtk.FILE_CHOOSER_ACTION_SAVE
            but = gtk.STOCK_SAVE
            title = 'Save image as ...'

        elif mode == 'open':
            act = gtk.FILE_CHOOSER_ACTION_OPEN
            but = gtk.STOCK_OPEN
            title = 'Open file ...'
            
        else: raise Exception("Mode must be 'save' or 'load'")
        
        fc = gtk.FileChooserDialog(
            title=title,
            parent=self.root,
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     but, gtk.RESPONSE_OK),
            action=act)
        fc.set_current_folder(os.getcwd())
        response = fc.run()

        
        if response == gtk.RESPONSE_OK:
            filepath = fc.get_filename()
            fc.destroy()
            action(filepath)
            
        elif response == gtk.RESPONSE_CANCEL:
            fc.destroy()
        
    def save_image(self, filepath):
        """Save an image as a FITS file

        :param filepath: File path.
        """
        self.tools.write_fits(
            filepath, self.image, fits_header=self.header, overwrite=True)

 
    def update_image(self, im):
        """Update displayed image

        :param im: New image to display
        """
        if self.dimx is None or self.dimy is None:
            return None

        if im is None:
            im = np.empty((self.dimx, self.dimy))
            im.fill(np.nan)

        self.image = im
        image = AstroImage.AstroImage(logger=self.logger)
        image.set_data(self.image)
        if self.wcs is not None:
            image.set_wcs(self.wcs)

        if self.fitsimage is not None:
            all_nans = np.all(np.isnan(im))
            if not self.init_set:
                if all_nans:
                    self.fitsimage.enable_autocuts('off')
                    self.fitsimage.set_autocut_params('minmax')
                
                self.fitsimage.enable_autozoom('off')
                self.fitsimage.set_autocenter('off')
                self.fitsimage.set_image(
                        image, raise_initialize_errors=False)
                if not all_nans:
                    self.fitsimage.enable_autocuts('on')
                    self._set_autocut_method_cb(None, self.autocut_index)
                
            else:
                self.fitsimage.transform(False, False, True)
                self.fitsimage.set_image(
                    image, raise_initialize_errors=False)
        
        if self.init_set:
            self.init_set = False
        

###########################################
### CLASS POPUPWINDOW #####################
###########################################
class PopupWindow(object):
    """Basic popup window frame"""

    def __init__(self, title='Popup', size=(300,300)):
        """Init Popup Window

        :param title: (Optional) Window title (default 'Popup').

        :param size: (Optional) Window size (default (300,300)).
        """
        self.w = gtk.Window()
        self.w.set_title(title)
        self.w.set_size_request(*size)
        

    def show(self):
        """Show popup window"""
        
        self.w.show_all()
        

###########################################
### CLASS HEADERWINDOW ####################
###########################################
class HeaderWindow(PopupWindow):
    """Header window.

    Display and manage a pyfits.Header instance
    """
    
    def __init__(self, header):
        """Init and construct header window.

        :param header: header to display. Must be a pyfits.Header instance.
        """
        PopupWindow.__init__(self, title='Header',
                             size=(300,500))
        
        self.header = header
        
        # fit results
        box = gtk.VBox()
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

        header_store = gtk.ListStore(str, str, str)
        header_tv = gtk.TreeView(header_store)
        key_render = gtk.CellRendererText()
        key_render.set_property('font', 'mono')
        key_render.set_property('size-points', 10)
        key_render.set_property('editable', True)
        key_render.set_property('weight', 800)
        
        value_render =  gtk.CellRendererText()
        value_render.set_property('font', 'mono')
        value_render.set_property('size-points', 10)
        value_render.set_property('editable', True)

        
        col_key = gtk.TreeViewColumn("Key", key_render, text=0)
        col_value = gtk.TreeViewColumn("Value", value_render, text=1)
        col_comment = gtk.TreeViewColumn("Comment", value_render, text=2)
      
        header_tv.append_column(col_key)
        header_tv.append_column(col_value)
        header_tv.append_column(col_comment)

        sw.add(header_tv)
        box.pack_start(sw, fill=True, expand=True)
        self.w.add(box)
        for i in range(len(self.header)):
            header_store.append([
                self.header.keys()[i],
                str(self.header[i]),
                self.header.comments[i]])
            
  
###########################################
### CLASS SPECTRUMWINDOW ##################
###########################################
class ZPlotWindow(PopupWindow):
    """Implement a window for plotting zaxis data."""


    is_spectrum = None
    _display_spectrum = None
    fitplugin = None
    
    def __init__(self, step, order, wavenumber, bunit):
        """Init and construct spectrum window.

        :param step: Step size [in nm]
        
        :param order: Aliasing Order
        
        :wavenumber: True if data axis is in wavenumber, False if in
          wavelength, None otherwise.

        :param bunit: Flux unit (string)
        """
        SIZE = (8,6)
        DPI = 75
        PopupWindow.__init__(self, title='Zdata',
                             size=(SIZE[0]*DPI,SIZE[1]*DPI))

        self.step = step
        self.order = order
        self.wavenumber = wavenumber
        if self.wavenumber is None:
            self.is_spectrum = False
            self._display_spectrum = False
            self.wavenumber = True
        else:
            self.is_spectrum = True
            
        self.bunit = bunit
        self.zmin_pix = None
        self.zmax_pix = None
        
        self.fig = Figure(figsize=SIZE, dpi=DPI, tight_layout=True)
        self.subplot = self.fig.add_subplot(111)


        # CREATE framebox
        framebox = gtk.VBox()
        # add figure canvas
        self.canvas = FigureCanvas(self.fig)  # a gtk.DrawingArea
        framebox.pack_start(self.canvas, fill=True, expand=True)

        # add buttons
        buttonsbar = gtk.HBox()
        if not self.is_spectrum:
            self.specbutton = gtk.Button('Spectrum')
            self.specbutton.connect('clicked', self._display_spectrum_cb)
            buttonsbar.pack_start(self.specbutton)
            
        framebox.pack_start(buttonsbar, fill=False, expand=False)
        

        # add fit plugin
        self.fitplugin = FitPlugin(step=self.step,
                                   order=self.order,
                                   update_cb=self.update,
                                   wavenumber=self.wavenumber)
        framebox.pack_start(self.fitplugin.get_frame(),
                            fill=False, expand=False)
        
        # add framebox
        self.w.add(framebox)
        self._start_plot_widgets()

    def _start_plot_widgets(self):
        if not self.is_spectrum:
            if not self._display_spectrum:
                self.span = SpanSelector(
                    self.subplot, self._span_select_cb,
                    'horizontal', useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'))
        

    def _display_spectrum_cb(self, c):
        self._display_spectrum = ~self._display_spectrum
        if self._display_spectrum:
            self.specbutton.set_label('Interferogram')
        else:
            self.specbutton.set_label('Spectrum')
        self.update()
            
    def _span_select_cb(self, zmin, zmax):
        if not self.is_spectrum:
            if not self._display_spectrum:
                self.zmin_pix = zmin
                self.zmax_pix = zmax
                self.update()
        
    def _get_zaxis(self, n):
        """Return the best zaxis based on known parameters

        :param n: Number of samples
        """
        if (self.step is not None
            and self.order is not None
            and self.is_spectrum):
            if self.wavenumber:
                return orb.utils.create_cm1_axis(
                    n, self.step, self.order)
            else:
                return orb.utils.create_nm_axis(
                    n, self.step, self.order)
        else:
            return np.arange(n)

    def update(self, zdata=None, zaxis=None):
        """Update plot

        :param zdata: Data to plot (Y)
        :param zaxis: X axis.
        """
        if zdata is not None:
            self.zdata = np.copy(zdata)
        else: zdata = self.zdata

        zdata[np.nonzero(zdata == 0.)] = np.nan
        self.subplot.cla()
        
        if zaxis is None:
            zaxis = self._get_zaxis(np.size(zdata))
            if self.is_spectrum:
                if self.wavenumber: self.subplot.set_xlabel('Wavenumber [cm-1]')
                else:  self.subplot.set_xlabel('Wavelength [nm]')
            else:
                if not self._display_spectrum:
                    self.subplot.set_xlabel('Step index')
                else:
                    if self.step is not None and self.order is not None:
                        self.subplot.set_xlabel('Wavenumber [cm-1]')
                    else:
                        self.subplot.set_xlabel('Channel index')
                
        if self.bunit is not None:
            if self.wavenumber is not None:
                self.subplot.set_ylabel(self.bunit)

        # compute spectrum
        if not self.is_spectrum:
            if self.zmin_pix is not None and self.zmax_pix is not None:
                interf = zdata[int(self.zmin_pix):int(self.zmax_pix)]
            else:
                interf = np.copy(zdata)
            #ext_phase = orb.utils.optimize_phase(interf, self.step, self.order, 0)

            if self.step is not None and self.order is not None:
                zdata = orb.utils.transform_interferogram(
                    interf, 1., 1., self.step, self.order, None, 0,
                    #ext_phase=ext_phase, wavenumber=True)
                    phase_correction=False, wavenumber=True,
                    low_order_correction=True)

                zaxis = orb.utils.create_cm1_axis(
                    spectrum.shape[0], self.step, self.order)
            else:
                zdata = orb.utils.raw_fft(interf)
                zaxis = np.arange(spectrum.shape[0])
                
                
        self.subplot.plot(zaxis, zdata, c='0.', lw=1.)


        if self.fitplugin is not None:
            if self.is_spectrum or self._display_spectrum:
                if self.fitplugin is not None:
                    self.fitplugin.set_spectrum(zdata)
                lines = self.fitplugin.get_fit_lines()
                if lines is not None:
                    for line in lines:
                        self.subplot.axvline(x=line, c='0.3', ls=':')
                fitted_vector = self.fitplugin.get_fitted_vector()
                if fitted_vector is not None:
                    self.subplot.plot(zaxis, fitted_vector)
            
        self.canvas.draw()
        self._start_plot_widgets()


###########################################
### CLASS FITPLUGIN #######################
###########################################
class FitPlugin(object):
    """Fitting plugin that can be added to the ZPlot module.
    """

    fit_lines = None
    wavenumber = None
    step = None
    order = None
    apod = None
    spectrum = None
    update_cb = None
    fitted_vector = None
    
    def __init__(self, step=None, order=None, apod=None,
                 update_cb=None, wavenumber=True):
        """Init FitPlugin class

        :param step: (Optional) Step size in nm (default None).

        :param order: (Optional) Folding order (default None).

        :param apod: (Optional) Apodization (default None).

        :param update_cb: (Optional) function to call when the visual
          object displaying the spectrum must be updated (the ZPlot
          window) (default None).

        :param wavenumber: (Optional) If True spectrum to fit is in
          wavenumber (default True).
        """

        self.fit_lines = None
        
        # FIT BOX
        frame = gtk.Frame('Spectrum Fit')
        framebox = gtk.HBox(spacing=2)
        fitbox = gtk.VBox(spacing=2)

        # Observation Params
        buttonbox = gtk.HBox()

        stepbox = gtk.VBox()
        self.wstep = gtk.Entry()
        self.wstep.set_width_chars(8)
        self.wstep.connect('changed', self._set_step_cb)
        wstep_label = gtk.Label('Step (nm)')
        wstep_label.set_max_width_chars(8)
        
        stepbox.pack_start(wstep_label)
        stepbox.pack_start(self.wstep)
        
        orderbox = gtk.VBox()
        self.worder = gtk.Entry()
        self.worder.set_width_chars(8)
        self.worder.connect('changed', self._set_order_cb)
        worder_label = gtk.Label('Order')
        worder_label.set_max_width_chars(8)
        orderbox.pack_start(worder_label)
        orderbox.pack_start(self.worder)

        apodbox = gtk.VBox()
        self.wapod = gtk.Entry()
        self.wapod.connect('changed', self._set_apod_cb)
        self.wapod.set_width_chars(8)
        wapod_label = gtk.Label('Apod')
        wapod_label.set_max_width_chars(8)
        apodbox.pack_start(wapod_label)
        apodbox.pack_start(self.wapod)

        # calibration map
        calibbox = gtk.VBox()
        caliblabel = gtk.Label('Calibration Map')
        
        wcalib = gtk.FileChooserButton("Choose a calibration map")
        wcalib.connect('file-set', self._get_calib_map_path_cb)
        calibbox.pack_start(caliblabel)
        calibbox.pack_start(wcalib)
        
        for box in (stepbox, orderbox, apodbox, calibbox):
            buttonbox.pack_start(box, fill=False, expand=False)
            
        fitbox.pack_start(buttonbox, fill=False, expand=False)

        # define choose line box
        linebox = gtk.HBox(spacing=2)
        wselectline = viewer2d.gtkw.Widgets.ComboBox()
        for line in self._get_lines_keys():
            wselectline.append_text(line)
        wselectline.set_index(0)
        self.line_name = self._get_lines_keys()[0]
        wselectline.add_callback('activated', self._set_line_name_cb)
        linebox.pack_start(wselectline.get_widget(), fill=True, expand=True)
        
        waddline = gtk.Button("+")
        waddline.connect('clicked', self._add_fit_line_cb)
        linebox.pack_start(waddline, fill=True, expand=True)
        wdelline = gtk.Button("-")
        wdelline.connect('clicked', self._del_fit_line_cb)
        linebox.pack_start(wdelline, fill=True, expand=True)

        fitbox.pack_start(linebox, fill=False, expand=False)

        # define velocity / redshift box
        velocitybox = gtk.HBox(spacing=2)
        velbox = gtk.VBox()
        wvelocity_label = gtk.Label('Velocity [km/s]')
        self.wvelocity = gtk.SpinButton(climb_rate=1., digits=1)
        self.wvelocity.set_range(-1e6,1e6)
        self.wvelocity.set_increments(10,100)
        self.wvelocity.connect('value-changed', self._update_velocity_cb)
        velbox.pack_start(wvelocity_label, fill=False, expand=False)
        velbox.pack_start(self.wvelocity, fill=False, expand=False)
        velocitybox.pack_start(velbox)

        redbox = gtk.VBox()
        wredshift_label = gtk.Label('Redshift')
        self.wredshift = gtk.Label('Redshift')
        self.wredshift = gtk.SpinButton(climb_rate=1., digits=4)
        self.wredshift.set_range(0, 1e8)
        self.wredshift.set_increments(0.01, 0.1)
        self.wredshift.connect('value-changed', self._update_velocity_cb)
        redbox.pack_start(wredshift_label, fill=False, expand=False)
        redbox.pack_start(self.wredshift, fill=False, expand=False)
        velocitybox.pack_start(redbox)
        fitbox.pack_start(velocitybox, fill=False, expand=False)

        self.fit_lines_velocity = 0.
        self.fit_lines_redshift = 0.

        # fit spectrum button
        wfit = gtk.Button("Fit Spectrum")
        wfit.connect('clicked', self._fit_lines_in_spectrum)
        fitbox.pack_start(wfit, fill=False, expand=False)

        framebox.pack_start(fitbox, fill=True, expand=False)
        # fit results
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

        self.fit_results_store = gtk.ListStore(
            str, str, str, str, str, str, str)
        wfit_results = gtk.TreeView(self.fit_results_store)
        renderer = gtk.CellRendererText()
        col_name = gtk.TreeViewColumn("Name", renderer, text=0)
        col_line = gtk.TreeViewColumn("Line", renderer, text=1)
        col_hei = gtk.TreeViewColumn("Height", renderer, text=2)
        col_amp = gtk.TreeViewColumn("Amplitude", renderer, text=3)
        col_vel = gtk.TreeViewColumn("Velocity", renderer, text=4)
        col_fwhm = gtk.TreeViewColumn("FWHM", renderer, text=5)
        col_snr = gtk.TreeViewColumn("SNR", renderer, text=6)
        wfit_results.append_column(col_name)
        wfit_results.append_column(col_line)
        wfit_results.append_column(col_hei)
        wfit_results.append_column(col_amp)
        wfit_results.append_column(col_vel)
        wfit_results.append_column(col_fwhm)
        wfit_results.append_column(col_snr)
        
        sw.add(wfit_results)
        framebox.pack_start(sw, fill=True, expand=True)

        frame.add(framebox)
        self.fitframe = frame
        self.set_step(step)
        self.set_order(order)
        self.set_apod(apod)
        self.wavenumber = wavenumber
        self.update_cb = update_cb
        self.fitted_vector = None


    def _get_calib_map_path_cb(self, w):
        """Callback to set the calibration laser map.

        :param w: Widget.
        """
        self._set_calib_map(w.get_filename())

    def _set_calib_map(self, file_path):
        """Set calibration laser map.

        :param file_path: Calibration laser map path.
        """
        self.calib_map = self.tools.read_fits(file_path)
        self.calib_map = orb.utils.interpolate_map(self.calib_map, self.dimx,
                                                   self.dimy)
        
    def _set_step_cb(self, w):
        """Callback to set the step size.
        
        :param w: Widget
        """
        self.step = float(w.get_text())
        
    def _set_order_cb(self, w):
        """Callback to set the folding order.
        
        :param w: Widget
        """
        self.order = int(float(w.get_text()))
        
    def _set_apod_cb(self, w):
        """Callback to set the step apodization function.
        
        :param w: Widget
        """
        if w.get_text() == 'None': self.apod = 1.0
        else: self.apod = float(w.get_text())

    def _fit_lines_in_spectrum(self, w):
        """Fit lines in the given spectrum
        
        :param w: Widget
        """
        if (self.fit_lines is not None
            and self.spectrum is not None
            and self.step is not None and self.order is not None):
            if len(self.fit_lines) > 0:
                fit_axis = self._get_axis()
                if self.wavenumber:
                    lines = orb.utils.nm2pix(fit_axis, self.get_fit_lines())
                lines = orb.utils.nm2pix(fit_axis, self.get_fit_lines())
                
                # remove lines that are not in the spectral range
                if self.wavenumber:
                    spectrum_range = orb.utils.cm12pix(
                        fit_axis, self._get_spectrum_range())
                else:
                    spectrum_range = orb.utils.nm2pix(
                        fit_axis, self._get_spectrum_range())

                lines = [line for line in lines
                         if (line > np.nanmin(spectrum_range)
                             and line < np.nanmax(spectrum_range))]

                if self.apod == 1.0: fmodel = 'sinc'
                else: fmodel = 'gaussian'

                # guess fwhm
                fwhm_guess = orb.utils.compute_line_fwhm(
                    self.spectrum.shape[0], self.step, self.order, apod_coeff=self.apod,
                    wavenumber=self.wavenumber)
                if self.wavenumber:
                    fwhm_guess_pix = orb.utils.cm12pix(
                        fit_axis, fit_axis[0] + fwhm_guess)
                else:
                    fwhm_guess_pix = orb.utils.nm2pix(
                        fit_axis, fit_axis[0] + fwhm_guess)

                fit_results = orb.utils.fit_lines_in_vector(
                    self.spectrum,
                    lines, fwhm_guess=fwhm_guess_pix,
                    signal_range=[np.nanmin(spectrum_range),
                                  np.nanmax(spectrum_range)],
                    return_fitted_vector=True, wavenumber=self.wavenumber,
                    observation_params=[self.step, self.order],
                    fmodel=fmodel)

                if 'lines-params-err' in fit_results:
                    self.fitted_vector = np.copy(fit_results['fitted-vector'])
                    self._update_plot()
                    
                    # print results
                    par = fit_results['lines-params']
                    par_err = fit_results['lines-params-err']
                    self.fit_results_store.clear()
                    for iline in range(len(self.fit_lines)):
                        store = np.zeros(7, dtype=float)
                        store[2] = par[iline][0]
                        store[3] = par[iline][1]

                        # convert velocity in km/s
                        if self.wavenumber:
                            pos = orb.utils.pix2cm1(fit_axis, par[iline][2])
                        else:
                            pos = orb.utils.pix2nm(fit_axis, par[iline][2])

                        store[4] = orb.utils.compute_radial_velocity(pos,
                            self.fit_lines[iline], wavenumber=self.wavenumber)

                        store[5] = par[iline][3] * abs(
                            fit_axis[1] - fit_axis[0])
                        store = ['{:.2e}'.format(store[i])
                                 for i in range(store.shape[0])]

                        if self.wavenumber:
                            store[0] = Lines().get_line_name(
                                orb.utils.cm12nm(self.fit_lines[iline]))
                        else:
                            store[0] = Lines().get_line_name(
                                self.fit_lines[iline])

                        store[1] = '{:.3f}'.format(self.fit_lines[iline])
                        store[6] = '{:.1f}'.format(
                            par[iline][1] / par_err[iline][1])

                        self.fit_results_store.append(store)

                    
    def _get_lines_keys(self):
        """Return lines keys."""
        keys = Lines().air_lines_nm.keys()
        keys.sort()
        keys.append('Filter lines')
        keys.append('Sky lines')
        return keys

    def _get_axis(self):
        """Return fit axis"""
        if self.order is not None and self.step is not None and self.spectrum is not None:
            if not self.wavenumber:
                return orb.utils.create_nm_axis(
                    self.spectrum.shape[0], self.step, self.order)
            else:
                return orb.utils.create_cm1_axis(
                    self.spectrum.shape[0], self.step, self.order)
        else: return None
            
    def _get_spectrum_range(self):
        """Return min and max wavelength of the spectrum"""
        axis = self._get_axis()
        if axis is not None:
            if self.spectrum is not None:
                nonans = np.nonzero(~np.isnan(self.spectrum))
            return np.nanmin(axis[nonans]), np.nanmax(axis[nonans])
        else:
            return None

        
    def _get_line_nm(self, line_name):
        """Return lines wavelength in nm

        :param line_name: Name of the lines
        """
        if self._get_spectrum_range() is None:
            return None
        else:
            if self.wavenumber:
                nm_max, nm_min = orb.utils.cm12nm(self._get_spectrum_range())
            else:
                nm_min, nm_max = self._get_spectrum_range()
                
        if line_name == 'Sky lines':
            delta_nm = orb.utils.compute_line_fwhm(
                    self.spectrum.shape[0], self.step, self.order, apod_coeff=self.apod,
                    wavenumber=False)
                
            return Lines().get_sky_lines(nm_min, nm_max, delta_nm)

        elif line_name == 'Filter lines':
            filter_lines = list()
            all_lines = Lines().air_lines_nm
        
            for line in all_lines.keys():
                line_nm = all_lines[line]
                
                if (line_nm > nm_min
                    and line_nm < nm_max):
                    filter_lines.append(line_nm)
            
            return filter_lines
            
        else:
            return Lines().get_line_nm(line_name)
 
    def _update_velocity_cb(self, w):
        """Callback to update velocity parameters (velocity and redshift)

        :param w: Widget.
        """
        self.fit_lines_velocity = self.wvelocity.get_value()
        self.fit_lines_redshift = self.wredshift.get_value()
        self._update_plot()
        
    def _add_fit_line_cb(self, w):
        """Add a line to the list of lines to fit.

        Action called by the '+' button.
        
        :param w: Widget
        """
        self._change_fit_lines(True)

        
    def _del_fit_line_cb(self, w):
        """Delete a line from the list of lines to fit.

        Action called by the '-' button.
        
        :param w: Widget
        """
        self._change_fit_lines(False)
        
    def _change_fit_lines(self, add):
        """Add or delete a fit line.

        :param add: If True line is added, if False line is deleted.
        """
        if self.fit_lines is None:
            self.fit_lines = []

        new_lines = list()
        new_lines = self._get_line_nm(self.line_name)

        if new_lines is None:
            return None
        
        if isinstance(new_lines, float):
            new_lines = list([new_lines])
        
        if self.wavenumber:
            new_lines = orb.utils.nm2cm1(new_lines)

        
        for new_line in new_lines:
            if add:
                if new_line not in self.fit_lines:
                    self.fit_lines.append(new_line)
            elif new_line in self.fit_lines:
                self.fit_lines.remove(new_line)
                
                
        # add lines to table
        self.fit_results_store.clear()
        for line in self.get_fit_lines():
            store = list(np.zeros(7, dtype=float))
            if self.wavenumber:
                store[0] = Lines().get_line_name(
                    orb.utils.cm12nm(line))
            else:
                store[0] = Lines().get_line_name(line)
                        
                store[1] = '{:.3f}'.format(line)
            self.fit_results_store.append(store)        

        self._update_plot()
        
    def _set_line_name_cb(self, w, idx):
        """Callback to set the line names

        :param w: Widget.
        :param idx: Index of the selected line.
        """
        self.line_name = self._get_lines_keys()[idx]


    def _update_plot(self):
        """Update plot in the ZPlot window if a callback function has
        been given.
        """
        if self.update_cb is not None:
            self.update_cb()

    def set_step(self, step):
        """Set step size.

        :param step: Step size in nm
        """
        if step is not None:
            self.wstep.set_text('{}'.format(step))

    def set_order(self, order):
        """Set order.

        :param order: Order.
        """
        if order is not None:
            self.worder.set_text('{}'.format(order))
            
    def set_apod(self, apod):
        """Set apodization.

        :param apod: Apodization function.
        """
        self.wapod.set_text('{}'.format(apod))

    def set_spectrum(self, spectrum):
        """Set spectrum to fit.

        :param spectrum: Spectrum.
        """
        update = False
        if self.spectrum is None:
            update = True
        elif np.all(np.isnan(self.spectrum)):
            update = True

        if spectrum is not None and self.spectrum is not None:
            nonans = np.nonzero(~np.isnan(spectrum))
            if np.any(self.spectrum[nonans] != spectrum[nonans]):
                update = True
        if update:
            self.spectrum = spectrum
            self.fitted_vector = None
    
    def get_frame(self):
        """Return the visual gtk.Frame object corresponding to the
        fitting plugin.
        """
        return self.fitframe

    def get_fit_lines(self):
        """Return the selected fit lines."""
        if self.fit_lines is not None:
            fit_lines = (np.array(self.fit_lines)
                         + np.array(orb.utils.line_shift(
                             self.fit_lines_velocity, self.fit_lines,
                             wavenumber=self.wavenumber)))
            if self.wavenumber:
                return fit_lines / (1. + self.fit_lines_redshift)
            else:
                return fit_lines * (1. + self.fit_lines_redshift)
        else:
            return None

    def get_fitted_vector(self):
        """Return the fitted spectrum."""
        return self.fitted_vector

###########################################
### CLASS REGIONS #########################
###########################################
class Regions(object):
    """Manage regions parameters and the visual objects (region menu)"""

    method = None
    methods = ['mean', 'sum', 'median']
    shapes = ['rectangle', 'circle']
    shape = None
    change_region_properties_cb = None
    
    def __init__(self, change_region_properties_cb=None):
        """Init Regions class.

        :param change_region_properties_cb: (Optional) method to call in the main
          window when region properties are changed (default None).
        """
        def new_checksubmenu(name, cb, args, first_radioitem, menuitems):
            _m = gtk.RadioMenuItem(first_radioitem, name)
            _m.connect('toggled', cb, args)
            if first_radioitem is None:
                first_radioitem = _m
                first_radioitem.set_active(True)
            menuitems.append(_m)
            return first_radioitem, menuitems

        self.change_region_properties_cb = change_region_properties_cb

        region_menu = gtk.Menu()
        self.region_mi = gtk.MenuItem('Region')
        self.region_mi.set_submenu(region_menu)

        # add method menu
        region_method_menu = gtk.Menu()
        region_method_mi = gtk.MenuItem('Method')
        region_method_mi.set_submenu(region_method_menu)

        first_radioitem = None
        menuitems = list()
        for imethod in self.methods:
            first_radioitem, menuitems = new_checksubmenu(
                imethod, self._set_method_cb, imethod, first_radioitem,
                menuitems)
        submenus = list()
        for menuitem in menuitems:
            submenus.append(menuitem)

        for submenu in submenus:
            region_method_menu.append(submenu)
                    
        region_menu.append(region_method_mi)
        
        self.method = 'sum'
        menuitems[self.get_method_index('sum')].set_active(True)

        # add shape menu
        region_shape_menu = gtk.Menu()
        region_shape_mi = gtk.MenuItem('Shape')
        region_shape_mi.set_submenu(region_shape_menu)

        first_radioitem = None
        menuitems = list()
        for ishape in self.shapes:
            first_radioitem, menuitems = new_checksubmenu(
                ishape, self._set_shape_cb, ishape, first_radioitem,
                menuitems)
        submenus = list()
        for menuitem in menuitems:
            submenus.append(menuitem)

        for submenu in submenus:
            region_shape_menu.append(submenu)
                    
        region_menu.append(region_shape_mi)
        
        self.shape = 'circle'
        menuitems[self.get_shape_index('circle')].set_active(True)

        
    def _set_method_cb(self, c, method):
        """callback method to select a new combining method.

        :param c: widget.
        :param method: Choosen method.
        """
        if c.get_active():
            self.set_method(method)
        
    def set_method(self, method):
        """Select a new combining method.

        :param method: Selected method.
        """
        self.method = method

    def get_method(self):
        """Get the selected combining method."""
        return self.method
  
    def get_menubar_item(self):
        """Return the region menubar item"""
        return self.region_mi

    def get_method_index(self, method):
        """Return combining method index.

        :param method: method.
        """
        for index in range(len(self.methods)):
            if method == self.methods[index]:
                return index
        return None

    def _set_shape_cb(self, c, shape):
        """Callback function to select region shape.

        :param c: Widget.
        :param shape: Selected shape.
        """
        if c.get_active():
            self.set_shape(shape)
        
    def set_shape(self, shape):
        """Select region shape.

        :param shape: region shape.
        """
        self.shape = shape
        if self.change_region_properties_cb is not None:
            self.change_region_properties_cb(self)

    def get_shape(self):
        """Return selected region shape"""
        return self.shape

    def get_shape_index(self, shape):
        """Return region shape index.

        :param shape: region shape.
        """
        for index in range(len(self.shapes)):
            if shape == self.shapes[index]:
                return index
        return None
    
