#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca> based on
#   example1_gtk.py created by Eric Jeschke (eric@naoj.org) as part of
#   Ginga (http://ejeschke.github.io/ginga/)
# File: visual.py

## Copyright (c) 2010-2014 Thomas Martin <thomas.martin.1@ulaval.ca>
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

# GINGA IMPORTS
from ginga.gtkw.ImageViewCanvasGtk import ImageViewCanvas
from ginga.gtkw.ImageViewCanvasTypesGtk import DrawingCanvas
import ginga.gtkw.Widgets
from ginga.gtkw import FileSelection
from ginga import AstroImage
from ginga.util import wcsmod
from ginga.util.wcsmod import AstropyWCS
wcsmod.use('astropy')

# ORB IMPORTS
from orb.core import Tools, Cube, HDFCube
import orb.utils

# OTHER IMPORTS
import gtk, gobject
import gtk.gdk
import numpy as np
import astropy.wcs as pywcs

# MATPLOTLIB GTK BACKEND
from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas
from matplotlib.figure import Figure

gobject.threads_init()

import socket
from threading import Thread, Event

###########################################
### CLASS BASEVIEWER ######################
###########################################

class BaseViewer(object):
    """This class implements a basic viewer based on GINGA for FITS/HDF5
    cubes created with ORB modules.

    .. seealso:: GINGA project: http://ejeschke.github.io/ginga/
    """

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
    
    def __init__(self, config_file_name='config.orb', no_log=True,
                 debug=False):

        """Init BaseViewer

        :param config_file_name: (Optional) Name of ORB config file
          (default 'config.orb').

        :param no_log: (Optional) If True no logfile will be created
          for ORB specific warnings, info and errors (default True).

        :param debug: (Optional) If True, all messages from Ginga are
          printed on stdout (default False).
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
        bm = fi.get_bindmap()
        bm.add_callback('mode-set', self._mode_change_cb)
        
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
        w.set_size_request(self.IMAGE_SIZE,self. IMAGE_SIZE)

        # BIG BOX
        bigbox = gtk.VBox(spacing=2)

        # IMAGE FRAME
        imageframe = gtk.Frame('Image')
        imageframebox = gtk.VBox()

        # imageoptionsbox
        imageoptionsbox = gtk.HBox()
        autocutbox = gtk.VBox()
        wautocuts = ginga.gtkw.Widgets.ComboBox()
        for name in self.autocut_methods:
            wautocuts.append_text(name)
        wautocuts.set_index(4)
        self.autocut_index = 4
        wautocuts.add_callback('activated', self._set_autocut_method_cb)
        wautocuts_label = gtk.Label('Scale')
        autocutbox.pack_start(wautocuts_label)
        autocutbox.pack_start(wautocuts.widget)
        imageoptionsbox.pack_start(autocutbox, fill=False, expand=False)

        indexbox = gtk.VBox()
        self.wimage_index = gtk.Adjustment(value=0, lower=0,
                                           upper=100, step_incr=1,
                                           page_incr=10)
        
        self.wimage_index.connect('value-changed', self._set_image_index_cb)
        self.index_label = gtk.Label('Image index')
        index_scale = gtk.HScale(self.wimage_index)
        index_scale.set_digits(0)
        index_scale.set_value_pos(gtk.POS_RIGHT)
        indexbox.pack_start(self.index_label)
        indexbox.pack_start(index_scale)
        imageoptionsbox.pack_start(indexbox, fill=True, expand=True)

        
        spacebox = gtk.HBox()
        imageoptionsbox.pack_start(spacebox, fill=True, expand=True)

        savebox = gtk.VBox()
        saveimage_label = gtk.Label('')
        saveimage = gtk.Button('Save Image')
        saveimage.connect('clicked', self._save_image_cb)
        savebox.pack_start(saveimage)
        savebox.pack_start(saveimage_label)
        imageoptionsbox.pack_start(savebox, fill=False, expand=False)
        
        imageframebox.pack_start(imageoptionsbox, fill=False, expand=False)
        imageframebox.pack_start(w)
        
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
        imagebox = gtk.HBox(spacing=2)
        imagebox.pack_start(imageframe, fill=True, expand=True)
        
        
        # MENU
        def new_submenu(name, cb):
            _m = gtk.MenuItem(name)
            _m.connect('activate', cb)
            return _m
            
        file_submenu = gtk.Menu()
        submenus = list()
        submenus.append(new_submenu('Open...',
                                    self._open_file_cb))
        submenus.append(new_submenu('Display header...',
                                    self._display_header_cb))
        
        for submenu in submenus:
            file_submenu.append(submenu)
        file_menu = gtk.MenuItem('File')
        file_menu.set_submenu(file_submenu)
        menu_bar = gtk.MenuBar()
        menu_bar.append(file_menu)
        
        # PACK BIGBOX
        bigbox.pack_start(menu_bar, fill=False, expand=False, padding=2)
        
        bigbox.pack_start(imagebox, fill=True, expand=True)

        # add plugins
        plugins = self._get_plugins()
        for plugin in plugins:
            bigbox.pack_start(plugin, fill=False, expand=False, padding=2)
        
        root.add(bigbox)

    def _get_plugins(self):
        return []

    def _reload_file(self):
        print 'Reloading {}'.format(self.filepath)
        self.load_file(self.filepath, reload=True)
        
    def _open_file_cb(self, c):
        """open-file callback

        Popup a File chooser dialog to open a new file
        """
        self.init_set = True
        self.pop_file_chooser_dialog(self.load_file)

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
        self.key_pressed = key

    def _key_released_cb(self, c, key):
        """key-released callback.
        
        :param c: Caller Instance
        :param key: key pressed
        """
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
        self.pop_file_chooser_dialog(self.save_image)

    def _mode_change_cb(self, c, mode, modetype):
        """change-mode callback

        Called when the interaction mode is changed (shift or ctrl pressed)

        Avoid drawing boxes while in special mode.

        :param c: Caller Instance

        :param mode: Interaction mode

        :param modetype: Mode type
        """
        print modetype
        self.mode = mode
        if mode in ['shift', 'ctrl']:
            self.canvas.enable_draw(False)
        else:
            self.canvas.enable_draw(True)

    def _set_autocut_method_cb(self, c, idx):
        """autocut-method-changed callback

        Changes the autocut method
        
        :param c: Caller Instance

        :param idx: Index of the new autocut method in
          self.autocut_methods list
        """
        self.autocut_index = idx
        self.fitsimage.set_autocut_params(self.autocut_methods[idx])        

    def _start_box_cb(self, c, button, data_x, data_y):
        """start-drawing-box callback

        Start drawing a box, delete old drawn boxes

        :param c: Caller instance

        :param button: button pressed while drawing (e.g. ctrl or shift)

        :param data_x: X position when mouse button pressed

        :param data_y: Y position when mouse button pressed
        """
        if self.mode not in ['ctrl', 'shift'] and button == 4:
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

        if self.mode not in ['ctrl', 'shift'] and button == 4:    
            self.xy_stop = (data_x + 1, data_y + 1)

            
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
                 
            
            if self.image is not None:
                self.image_region = self.image[x_range[0]:x_range[1],
                                               y_range[0]:y_range[1]]
                
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

                if len(zdata.shape) == 3:
                    zdata = np.nansum(np.nansum(zdata, axis=0), axis=0)


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
        
    def load_file(self, filepath, reload=False):
        """Load the file to display. Can be a FITS or HDF5 cube.

        :param filepath: Path to the file.
        """
        if os.path.splitext(filepath)[-1] in ['.fits']:
            self.hdf5 = False
            hdu = self.tools.read_fits(filepath, return_hdu_only=True)
            cube_size = float(hdu[0].header['NAXIS1']) * float(hdu[0].header['NAXIS2']) * float(hdu[0].header['NAXIS3']) * 4 / 1e9
            
            if cube_size > self.MAX_CUBE_SIZE:
                raise Exception('Cube size is too large: {} Go > {} Go'.format(cube_size, self.MAX_CUBE_SIZE))
                
            self.cube, self.header = self.tools.read_fits(filepath,
                                                          return_header=True,
                                                          memmap=True,
                                                          dtype=np.float32)
            
            self.filepath = filepath

        elif os.path.splitext(filepath)[-1] in ['.hdf5']:
            self.hdf5 = True
            self.cube = HDFCube(filepath)
            self.header = self.cube.get_cube_header()
            self.filepath = filepath

        else:
            self._print_error('File must be a FITS of HDF5 cube')

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
            if val < 0 or val >= self.cube.dimz:
                val = 0
            image_data = self.cube[:,:,val]    
            self.update_image(image_data)
            

        # SET ZAXIS
        self.wimage_index.set_upper(self.dimz-1)
        if self.header is not None:
            if 'ORDER' in self.header:
                self.order = self.header['ORDER']
            if 'STEP' in self.header:
                self.step = self.header['STEP']
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


    def pop_file_chooser_dialog(self, action):
        """Pop a a file chooser dialog

        :param action: Method launched when file path has been
          defined.
        """
        fc = gtk.FileChooserDialog(
            title='Save Image as ...',
            parent=self.root,
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     gtk.STOCK_SAVE, gtk.RESPONSE_OK),
            action=gtk.FILE_CHOOSER_ACTION_SAVE)
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
    """Implement a window for plotting zaxis data.
    """
    
    def __init__(self, step, order, wavenumber, bunit):
        """Init and construct spectrum window.

        :param step: Step size [in nm]
        
        :param order: Aliasing Order
        
        :wavenumber: True if data axis is in wavenumber, False if in
          wavelength, None otherwise.

        :param bunit: Flux unit (string)
        """
        SIZE = (8,3)
        DPI = 75
        PopupWindow.__init__(self, title='Zdata',
                             size=(SIZE[0]*DPI,SIZE[1]*DPI))

        # create zaxis
        self.step = step
        self.order = order
        self.wavenumber = wavenumber
        self.bunit = bunit

        self.fig = Figure(figsize=SIZE, dpi=DPI, tight_layout=True)
        self.subplot = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvas(self.fig)  # a gtk.DrawingArea
        self.w.add(self.canvas)

    def _get_zaxis(self, n):
        """Return the best zaxis based on known parameters

        :param n: Number of samples
        """
        if (self.step is not None
            and self.order is not None
            and self.wavenumber is not None):
            if self.wavenumber:
                return orb.utils.create_cm1_axis(
                    n, self.step, self.order)
            else:
                return orb.utils.create_nm_axis(
                    n, self.step, self.order)
        else:
            return np.arange(n)

    def update(self, zdata, zaxis=None):
        """Update plot

        :param zdata: Data to plot (Y)
        :param zaxis: X axis.
        """
        self.subplot.cla()
        if zaxis is None:
            zaxis = self._get_zaxis(np.size(zdata))
            if self.wavenumber is not None:
                if self.wavenumber: self.subplot.set_xlabel('Wavenumber [cm-1]')
                else:  self.subplot.set_xlabel('Wavelength [nm]')
        if self.bunit is not None:
            self.subplot.set_ylabel(self.bunit)
        self.subplot.plot(zaxis, zdata, c='0.', lw=1.)
        self.canvas.draw()


