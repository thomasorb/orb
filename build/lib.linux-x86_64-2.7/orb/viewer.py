#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca> 
# File: visual.py

## Copyright (c) 2010-2016 Thomas Martin <thomas.martin.1@ulaval.ca>
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
import traceback
import warnings

# ORB IMPORTS
from core import Tools, HDFCube, Lines
import utils.spectrum
import utils.fft
import utils.stats
import fit
import orb.cutils

# OTHER IMPORTS
import gtk, gobject
import gtk.gdk
import numpy as np
import astropy.wcs as pywcs
import math
import bottleneck as bn

# MATPLOTLIB GTK BACKEND
import matplotlib
matplotlib.use('GTKAgg')
from matplotlib.colorbar import ColorbarBase
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, RectangleSelector, EllipseSelector, AxesWidget
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

import pylab as pl

gobject.threads_init()


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
    axis_corr = None

    spectrum_window = None
    
    _plugins_postload_calls = list()

    DRAW_COLOR_DEFAULT = 'green'
    
    BAD_WCS_FLAG = False

    WINDOW_SIZE = (550,500)

    MAX_CUBE_SIZE = 4. # Max cube size in Go
    MAX_CUBE_SECTION_SIZE = 1. # max section size to load in Go
 
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
    
        
        self.tools = Tools()
        self.tools._no_log = no_log
        self.tools.config_file_name = config_file_name
        self.init_set = True
        self.spectrum_channel = 1
        self.wcs_rotation = float(
            self.tools._get_config_parameter('WCS_ROTATION'))
        self.fov = float(
            self.tools._get_config_parameter('FIELD_OF_VIEW_1'))
           
           
        # Define GTK GUI
        root = gtk.Window(gtk.WINDOW_TOPLEVEL)
        root.set_title("ORB Viewer")
        root.set_border_width(2)
        root.set_size_request(*self.WINDOW_SIZE)
        root.connect("delete_event", lambda w, e: self._quit_cb(w))
        self.root = root
        
        # define ImageCanvas
        self.regions = Regions()
        self.canvas = ImageCanvas(self.regions)
        self.canvas.connect('on-select-region', self._on_select_region_cb)
        self.canvas.connect('on-move', self._on_move_cb)
        self.autocut_methods = self.canvas.get_autocut_methods()

        # VIEWER_HBOX
        viewer_hbox = gtk.HBox(spacing=2)
    
        # VIEWER_VBOX
        viewer_vbox = gtk.VBox(spacing=2)

        # IMAGE FRAME
        imageframe = gtk.Frame('Image')
        imageframebox = gtk.VBox()

        # imageoptionsbox
        imageoptionsbox = gtk.HBox()
        
        # autocut
        autocutbox = gtk.VBox()
        wautocuts = gtk.combo_box_new_text()
        for name in self.autocut_methods:
            wautocuts.append_text(name)
        wautocuts.set_active(4)
        self.autocut_index = 4
        wautocuts.connect('changed', self._set_autocut_method_cb)
        wautocuts_label = gtk.Label('Scale')
        autocutbox.pack_start(wautocuts_label)
        autocutbox.pack_start(wautocuts)
        imageoptionsbox.pack_start(autocutbox, fill=False, expand=False)

        # colormap 
        colormapbox = gtk.VBox()
        wcolormap = gtk.combo_box_new_text()
        for name in self.canvas.get_cmaps():
            wcolormap.append_text(name)
        wcolormap.set_active(0)
        wcolormap.connect('changed', self._set_colormap_cb)
        wcolormap_label = gtk.Label('Color Map')
        colormapbox.pack_start(wcolormap_label)
        colormapbox.pack_start(wcolormap)
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
        imageframebox.pack_start(self.canvas.get_widget(), fill=True, expand=True)
        
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

        # append region submenu
        menu_bar.append(self.regions.get_menubar_item())
        
        # PACK VIEWER_VBOX
        viewer_vbox.pack_start(menu_bar, fill=False, expand=False, padding=2)
        
        viewer_vbox.pack_start(imagebox, fill=True, expand=True)

        # add plugins
        plugins = self._get_vplugins()
        for plugin in plugins:
            viewer_hbox.pack_start(plugin, fill=False, expand=False, padding=2)

        # PACK VIEWER_HBOX
        viewer_hbox.pack_start(viewer_vbox, fill=True, expand=True)

        # add plugins
        plugins = self._get_hplugins()
        for plugin in plugins:
            viewer_hbox.pack_start(plugin, fill=False, expand=False, padding=2)

        # ADD to ROOT
        root.add(viewer_hbox)
        self.canvas.connect_all() # muy important
        root.show_all()

    def _print_traceback(self):
        """Print a traceback"""
        print traceback.format_exc()


    def _get_vplugins(self):
        """A list of plugins (returned as a list of gtk.Box instance)
        may be defined and incerted here. It will be packed vertically
        at the bottom of the basic view.
        """
        return []
    
    def _get_hplugins(self):
        """A list of plugins (returned as a list of gtk.Box instance)
        may be defined and incerted here. It will be packed horizontally
        at the left of the basic view.
        """
        return []
    

    def _reload_file(self):
        """Reload a file and update the displayed data. The file might
        have a new size.
        """
        print ' > Reloading {}'.format(self.filepath)
        self.load_file(self.filepath, reload=True)


    def _change_region_properties_cb(self, regions):
        #self.canvas.set_drawtype(
        #    regions.get_shape(), color=self.DRAW_COLOR_DEFAULT, alpha=1)
        pass

    def _set_colormap_cb(self, c):
        self.canvas.set_cmap(self.canvas.get_cmaps()[c.get_active()])
        
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
                conv = '{:.2f} nm'.format(utils.spectrum.cm12nm(zval))
            else:
                unit = 'nm'
                conv = '{:.2f} cm-1'.format(utils.spectrum.nm2cm1(zval))
            self.index_label.set_text('{:.2f} {} ({})'.format(
                zval, unit, conv))
            
        self.update_image(self.cube[:,:,val])
    
    def _save_image_cb(self, c):
        """save-image callback

        Pop a file chooser dialog to save the displayed Image

        :param c: Caller Instance
        """
        self.pop_file_chooser_dialog(self.save_image, mode='save')


    def _set_autocut_method_cb(self, c, index=None):
        """autocut-method-changed callback

        Changes the autocut method
        
        :param c: Caller Instance

        :param index: (Optional) Index
        """
        if index is None:
            index = c.get_active()
        
        self.autocut_index = index
        self.canvas.set_autocut_method(self.autocut_methods[index])


    def _on_select_region_cb(self, region):
        """on select region callback

        :param region: Region instance
        """
        
        self.image_region = self.canvas.image[region.get_selected_pixels()]
        x_range = region.x_range
        y_range = region.y_range
        

        self.mean.set_text('{:.3e}'.format(
            np.nanmean(self.image_region)))
        self.median.set_text('{:.3e}'.format(
            utils.stats.robust_median(self.image_region)))
        self.std.set_text('{:.3e}'.format(
            np.nanstd(self.image_region)))
        self.sum.set_text('{:.3e}'.format(
            np.nansum(self.image_region)))
        self.surf.set_text('{:.2e}'.format(
                self.image_region.size))


        # plot spectrum
        if self.dimz > 1:
            if self.spectrum_window is None:
                self.spectrum_window = ZPlotWindow(
                    self.step, self.order,
                    self.wavenumber, self.bunit,
                    axis_corr=self.axis_corr)
                self.spectrum_window.show()
            elif not self.spectrum_window.w.get_property('visible'):
                self.spectrum_window = ZPlotWindow(
                    self.step, self.order,
                    self.wavenumber, self.bunit,
                     axis_corr=self.axis_corr)
                self.spectrum_window.show()

            if self.hdf5:
                self.cube._silent_load = True

            
            zdata = self.cube[int(x_range[0]):int(x_range[1]),
                              int(y_range[0]):int(y_range[1]), :]
        
            if self.regions.get_shape() == 'circle':
                all_vectors = list()
                for ik in range(len(region.get_selected_pixels()[0])):
                    all_vectors.append(zdata[
                        region.get_selected_pixels()[0][ik] - x_range[0],
                        region.get_selected_pixels()[1][ik] - y_range[0],:])
                zdata = np.array(all_vectors)
            
            if len(zdata.shape) == 3:
                zdata = zdata.reshape((zdata.shape[0]*zdata.shape[1], zdata.shape[2]))
            
            if len(zdata.shape) == 2:
                if self.regions.get_method() == 'sum':
                    zdata = np.nansum(zdata, axis=0)
                elif self.regions.get_method() == 'mean':
                    zdata = np.nanmean(zdata, axis=0)
                elif self.regions.get_method() == 'median':
                    zdata = np.nanmedian(zdata, axis=0)
                else: raise Exception('Method error')

            try:
                self.spectrum_window.update(zdata)
            except Exception, e:
                print e
                
        
    def _on_move_cb(self, data_x, data_y):
        """mouse-motion callback

        Called when the mouse is moving on the image display.
        
        Display some informations on the mouse position.
    
        :param data_x: X position of the mouse on the image display.

        :param data_y: Y position of the mouse on the image display.
        """
        # Get the value under the data coordinates
        try:
            value = self.canvas.image[int(data_x + 0.5),
                                      int(data_y + 0.5)]
            
        except Exception:
            value = None

        if data_x is None or  data_y is None:
            return
        
        fits_x, fits_y = data_x, data_y

        # Calculate WCS RA/DEC
        try:
            ra, dec = self.wcs.wcs.wcs_pix2world(fits_x, fits_y, 0)
            ra_txt = '{:.0f}:{:.0f}:{:.1f}'.format(*utils.astrometry.deg2ra(ra))
            dec_txt = '{:.0f}:{:.0f}:{:.1f}'.format(*utils.astrometry.deg2dec(dec))
            
        except Exception as e:
            #print "Bad coordinate conversion: %s" % (str(e))
            ra_txt  = 'BAD WCS'
            dec_txt = 'BAD WCS'

        # hack: x/y inverted
        self.ra.set_text('{}'.format(ra_txt))
        self.dec.set_text('{}'.format(dec_txt))
        self.x.set_text('{:.2f}'.format(fits_x))
        self.y.set_text('{:.2f}'.format(fits_y))
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

    def _add_plugins_postload_call(self, call):
        """Add a call to the list of postload calls

        :param call: Call to be added (a function pointer)
        """
        self._plugins_postload_calls.append(call)

    def _plugins_postload_call(self):
        """Call all plugins after data has been loaded"""

        for icall in self._plugins_postload_calls:
            try:
                icall()
            except Exception, e:
                warnings.warn('> Error during plugins postload call: {}'.format(e))
                self._print_traceback()
                
            
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
        
        wcs_header = self.header               
        self.wcs = pywcs.WCS(wcs_header, naxis=2)
        
        try:
            ra, dec = self.wcs.wcs_pix2world(-1, -1, 0)
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

            if 'AXISCORR' in self.header:
                self.axis_corr = self.header['AXISCORR']
            else:
                self.axis_corr = 1.
                
            if 'CUNIT3' in self.header:
                if 'nm' in self.header['CUNIT3']:
                    self.wavenumber = False
                elif 'cm-1' in self.header['CUNIT3']:
                    self.wavenumber = True
                
            if (self.order is not None
                and self.step is not None
                and self.wavenumber is not None):
                if self.wavenumber:
                    self.zaxis = utils.spectrum.create_cm1_axis(
                        self.dimz, self.step, self.order,
                        corr=self.axis_corr)
                else:
                    self.zaxis = utils.spectrum.create_nm_axis(
                        self.dimz, self.step, self.order,
                        corr=self.axis_corr)

            if 'BUNIT' in self.header:
                if 'erg/cm^2/s/A' in self.header.comments['BUNIT']:
                    self.bunit = 'erg/cm^2/s/A'
                else:
                    self.bunit = self.header.comments['BUNIT']
                    
        self.root.set_title(filepath)
        self._plugins_postload_call()


    def pop_file_chooser_dialog(self, action, mode='save'):
        """Pop a a file chooser dialog

        :param action: Method launched when file path has been
          defined.
        """
        if mode == 'save':
            act = gtk.FILE_CHOOSER_ACTION_SAVE
            but = gtk.STOCK_SAVE
            title = 'Save image as...'

        elif mode == 'open':
            act = gtk.FILE_CHOOSER_ACTION_OPEN
            but = gtk.STOCK_OPEN
            title = 'Open file'
            
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
            
        else:
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

        self.canvas.set_image(im)
        
        if self.wcs is not None:
            self.canvas.set_wcs(self.wcs)
            
        all_nans = np.all(np.isnan(im))
        if not self.init_set:

            self.canvas.properties['autozoom'] = False
            self.canvas.properties['autocenter'] = False

            if not all_nans:
                self.canvas.properties['autocut'] = True
                self._set_autocut_method_cb(None, self.autocut_index)

        else:
            #self.canvas.transform(False, False, True)
            pass
        
        if self.init_set:
            self.init_set = False

###########################################
### CLASS AUTOCUTS ########################
###########################################
class AutoCuts(object):

    autocut_methods = ['min-max', 'rhist', '0.99', '0.95', '0.90', '0.85', '0.80']
    autocut_method = None
    im = None

    def __init__(self):
        self.autocut_method = 'rhist'

    def get_autocut_methods(self):
        return self.autocut_methods

    def set_autocut_method(self, autocut_method):
        if autocut_method in self.autocut_methods:
            self.autocut_method = autocut_method
        else:
            raise Exception('Bad autocut method ({}): must be in {}'.format(
                autocut_method, self.autocut_methods))

    def random_image_reduction(self, im):
        PROP = 0.01
        nb = int(im.size * PROP)
        rndx = np.random.randint(
            im.shape[0], size=nb)
        rndy = np.random.randint(
            im.shape[1], size=nb)
        im = np.copy(im[rndx, rndy])
        return im
        
        
    def get_vlim(self, im):
        distrib = self.random_image_reduction(im)
        
        if self.autocut_method == 'min-max':
            return bn.nanmin(im), bn.nanmax(im)
        elif self.autocut_method == 'rhist':
            return self._rhist_cut(distrib)
        else:
            try:
                coeff = float(self.autocut_method)
            except Exception, e:
                self._print_error('Bad autocut method: {}'.format(e))
            vmin = orb.cutils.part_value(distrib, 1.-coeff)
            vmax = orb.cutils.part_value(distrib, coeff)
            return vmin, vmax

    def _rhist_cut(self, distrib):
        
        distrib = utils.stats.sigmacut(distrib, sigma=2.5)
        
        return bn.nanmin(distrib), bn.nanmax(distrib)
        

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

    subplot = None
    simple_mode = False
    is_spectrum = None
    _display_spectrum = None
    fitplugin = None
    _kzdata = None # keeped zdata
    _substract = False
    
    def __init__(self, step, order, wavenumber, bunit,
                 axis_corr=1.,
                 title='Zdata', simple=False):
        """Init and construct spectrum window.

        :param step: Step size [in nm]
        
        :param order: Aliasing Order
        
        :wavenumber: True if data axis is in wavenumber, False if in
          wavelength, None otherwise.

        :param bunit: Flux unit (string)

        :param axis_corr: (Optional) Wave axis correction coefficient
          (default 1.)

        :param title: (Optional) Window title (default Zdata)

        :param simple: (Optional) If True, window display only a plot
          (no fit plugin) (default False).
        """
        SIZE = (8,6)
        DPI = 75
        PopupWindow.__init__(self, title=title,
                             size=(SIZE[0]*DPI,SIZE[1]*DPI))
        self.simple_mode = simple
        self.step = step
        self.order = order
        self.axis_corr = axis_corr
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
        self.twin_plot = self.fig.add_subplot(111)
        self.subplot = self.twin_plot.twinx()


        # CREATE framebox
        framebox = gtk.VBox()
        # add figure canvas
        self.canvas = FigureCanvas(self.fig)  # a gtk.DrawingArea
        framebox.pack_start(self.canvas, fill=True, expand=True)

        # add buttons

        if not self.simple_mode:
            buttonsbar = gtk.HBox()
            if not self.is_spectrum:
                self.specbutton = gtk.Button('Show spectrum')
                self.specbutton.connect('clicked', self._display_spectrum_cb)
                buttonsbar.pack_start(self.specbutton)

            keepbutton = gtk.Button('Keep')
            keepbutton.connect('clicked', self._keep_data_cb)
            buttonsbar.pack_start(keepbutton)

            subbutton = gtk.Button('Substract')
            subbutton.connect('clicked', self._substract_data_cb)
            buttonsbar.pack_start(subbutton)

            clabutton = gtk.Button('Clear')
            clabutton.connect('clicked', self._clear_data_cb)
            buttonsbar.pack_start(clabutton)

            uzobutton = gtk.Button('Unzoom')
            uzobutton.connect('clicked', self._unzoom_cb)
            buttonsbar.pack_start(uzobutton)

            savbutton = gtk.Button('Save')
            savbutton.connect('clicked', self._save_data_cb)
            buttonsbar.pack_start(savbutton)

            framebox.pack_start(buttonsbar, fill=False, expand=False)
        

            # add fit plugin
            self.fitplugin = FitPlugin(step=self.step,
                                       order=self.order,
                                       axis_corr=self.axis_corr,
                                       update_cb=self.update,
                                       wavenumber=self.wavenumber,
                                       parent=self)
            framebox.pack_start(self.fitplugin.get_frame(),
                                fill=False, expand=False)
        
        # add framebox
        self.w.add(framebox)
        self._start_plot_widgets()

    def _start_plot_widgets(self):
        if not self.is_spectrum:
            if not self._display_spectrum:
                if not self.simple_mode:
                    self.span = SpanSelector(
                        self.subplot, self._span_select_cb,
                        'horizontal', useblit=False,
                        rectprops=dict(alpha=0.5, facecolor='red'))
                
        else:
            if not self.simple_mode:
                self.span_fit = SpanSelector(
                    self.subplot, self._span_fit_cb,
                    'horizontal', useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='blue'))

        self.zoom = RectangleSelector(
            self.subplot, self._zoom_cb,
            useblit=False, button=3)

    def _display_spectrum_cb(self, c):
        self._display_spectrum = ~self._display_spectrum
        if self._display_spectrum:
            self.specbutton.set_label('Show interferogram')
        else:
            self.specbutton.set_label('Show spectrum')
        self.update(reset_axis=True)

    def _unzoom_cb(self, c):
        self.update(reset_axis=True)


    def _save_data_cb(self, c):
        fc = gtk.FileChooserDialog(
            title='Save data as...',
            parent=self.w,
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                     gtk.STOCK_SAVE, gtk.RESPONSE_OK),
            action= gtk.FILE_CHOOSER_ACTION_SAVE)
        fc.set_current_folder(os.getcwd())
        
        response = fc.run()

        
        if response == gtk.RESPONSE_OK:
            filepath = fc.get_filename()
            fc.destroy()
            Tools().write_fits(filepath, self.zdata, overwrite=True)
        
        else:
            fc.destroy()
        
        
    def _zoom_cb(self, eclick, erelease):
        xlim = (min(eclick.xdata, erelease.xdata),
                max(eclick.xdata, erelease.xdata))
        ylim = (min(eclick.ydata, erelease.ydata),
                max(eclick.ydata, erelease.ydata))
        self.subplot.set_xlim(xlim)
        self.subplot.set_ylim(ylim)
        self.canvas.draw()


    def _keep_data_cb(self, c):
        if self.zdata is not None:
            self._kzdata = np.copy(self.zdata)
        self._substract = False
        self.update()

    def _substract_data_cb(self, c):
        self._keep_data_cb(c)
        self._substract = True
        self.update()

    def _clear_data_cb(self, c):
        self._substract = False
        self._kzdata = None
        self.update()
        
            
    def _span_select_cb(self, zmin, zmax):
        if not self.is_spectrum:
            if not self._display_spectrum:
                self.zmin_pix = zmin
                self.zmax_pix = zmax
                self.update()

    def _span_fit_cb(self, zmin, zmax):
        if self.is_spectrum:
            self.fitplugin.set_fitrange(zmin, zmax)
        
    def _get_zaxis(self, n):
        """Return the best zaxis based on known parameters

        :param n: Number of samples
        """
        if (self.step is not None
            and self.order is not None
            and self.is_spectrum):
            if self.wavenumber:
                return utils.spectrum.create_cm1_axis(
                    n, self.step, self.order, corr=self.axis_corr)
            else:
                return utils.spectrum.create_nm_axis(
                    n, self.step, self.order, corr=self.axis_corr)
        else:
            return np.arange(n)

    def update(self, zdata=None, zaxis=None, ylabel=None, reset_axis=False):
        """Update plot

        :param zdata: Data to plot (Y)
        :param zaxis: X axis.
        """
        if zdata is not None:
            self.zdata = np.copy(zdata)
            xlim = None
            ylim = None
        else:
            zdata = self.zdata
            if not reset_axis:
                xlim = self.subplot.get_xlim()
                ylim = self.subplot.get_ylim()
            else:
                xlim = None ; ylim = None
            
        if self._kzdata is not None:
            kzdata = np.copy(self._kzdata)

        #zdata[np.nonzero(zdata == 0.)] = np.nan
        self.subplot.cla()
        if self.twin_plot is not None:
            self.twin_plot.cla()
        
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

        zdata_phase = None
        
        # compute spectrum
        if not self.is_spectrum and self._display_spectrum:
            if self.zmin_pix is not None and self.zmax_pix is not None:
                interf = zdata[int(self.zmin_pix):int(self.zmax_pix)]
                if self._kzdata is not None:
                    kinterf = self._kzdata[int(self.zmin_pix):int(self.zmax_pix)]
            else:
                interf = np.copy(zdata)
                if self._kzdata is not None:
                    kinterf = np.copy(kzdata)
                    
            if self.step is not None and self.order is not None:
                zdata = utils.fft.transform_interferogram(
                    interf, 1., 1., self.step, self.order, None, 0,
                    phase_correction=False, wavenumber=True,
                    low_order_correction=True)
                zdata_phase = utils.fft.transform_interferogram(
                    interf, 1., 1., self.step, self.order, None, 0,
                    phase_correction=False, wavenumber=True,
                    low_order_correction=True, return_phase=True)
                if self._kzdata is not None:
                    kzdata = utils.fft.transform_interferogram(
                        kinterf, 1., 1., self.step, self.order, None, 0,
                        phase_correction=False, wavenumber=True,
                        low_order_correction=True)

                zaxis = utils.spectrum.create_cm1_axis(
                    interf.shape[0], self.step, self.order)
            else:
                zdata = utils.fft.raw_fft(interf)
                zdata_phase = utils.fft.raw_fft(interf, return_phase=True)
                if self._kzdata is not None:
                    kzdata = utils.fft.raw_fft(kinterf)
                zaxis = np.arange(interf.shape[0])

        # plot data
        if self._kzdata is not None:
            if self._substract:
                zdata = np.copy(zdata) - kzdata
            else:
                self.subplot.plot(zaxis, kzdata, c='red', lw=1.)
                
        if zdata_phase is not None:
            
            #self.subplot.plot(zaxis, zdata_phase, c='green', lw=1.)
            self.twin_plot.plot(zaxis, zdata_phase, c='green', lw=1.)
            self.twin_plot.set_ylabel('Phase')
        elif self.twin_plot is not None:
            self.twin_plot.cla()
        
            
        self.subplot.plot(zaxis, zdata, c='0.', lw=1.)
        
        if ylabel is not None:
            self.subplot.set_ylabel(ylabel)
        

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

        if xlim is not None:
            self.subplot.set_xlim(xlim)
            self.subplot.set_ylim(ylim)
        self.canvas.draw()
        self._start_plot_widgets()


###########################################
### CLASS FITPLUGIN #######################
###########################################
class FitPlugin(object):
    """Fitting plugin that can be added to the ZPlot module.
    """

    line_models = ['auto', 'sinc', 'gaussian', 'sinc2']
    fit_lines = None
    wavenumber = None
    step = None
    order = None
    axis_corr = None
    apod = None
    spectrum = None
    update_cb = None
    fitted_vector = None
    fitrange = None
    parent = None
    
    def __init__(self, step=None, order=None, axis_corr=1., apod=None,
                 update_cb=None, wavenumber=True, parent=None):
        """Init FitPlugin class

        :param step: (Optional) Step size in nm (default None).

        :param order: (Optional) Folding order (default None).

        :param axis_corr: (Optional) Wave axis correction coefficient
          (default 1.)

        :param apod: (Optional) Apodization (default None).

        :param update_cb: (Optional) function to call when the visual
          object displaying the spectrum must be updated (e.g. ZPlot
          window) (default None).

        :param wavenumber: (Optional) If True spectrum to fit is in
          wavenumber (default True).

        :param parent: (Optional) Parent class (e.g. ZPlot Window)
        """

        self.fit_lines = None

        if parent is not None: self.parent = parent
        self.axis_corr = axis_corr
        
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
        wselectline = gtk.combo_box_new_text()
        for line in self._get_lines_keys():
            wselectline.append_text(line)
        wselectline.set_active(0)
        self.line_name = self._get_lines_keys()[0]
        wselectline.connect('changed', self._set_line_name_cb)
        linebox.pack_start(wselectline, fill=True, expand=True)
        
        waddline = gtk.Button("+")
        waddline.connect('clicked', self._add_fit_line_cb)
        linebox.pack_start(waddline, fill=True, expand=True)
        wdelline = gtk.Button("-")
        wdelline.connect('clicked', self._del_fit_line_cb)
        linebox.pack_start(wdelline, fill=True, expand=True)

        wselectlinemodel = gtk.combo_box_new_text()
        for linemodel in self.line_models:
            wselectlinemodel.append_text(linemodel)
        wselectlinemodel.set_active(0)
        self.line_model = self.line_models[0]
        wselectlinemodel.connect('changed', self._set_line_model_cb)
        linebox.pack_start(wselectlinemodel, fill=True, expand=True)

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
        self.calib_map = utils.image.interpolate_map(
            self.calib_map, self.dimx, self.dimy)
        
    def _set_step_cb(self, w):
        """Callback to set the step size.
        
        :param w: Widget
        """
        self.set_step(w.get_text())
        
    def _set_order_cb(self, w):
        """Callback to set the folding order.
        
        :param w: Widget
        """
        self.set_order(w.get_text())
        
    def _set_apod_cb(self, w):
        """Callback to set the step apodization function.
        
        :param w: Widget
        """
        if w.get_text() == 'None': self.set_apod('1.0')
        else: self.set_apod(w.get_text())

    def _fit_lines_in_spectrum(self, w):
        """Fit lines in the given spectrum
        
        :param w: Widget
        """
        if (self.fit_lines is not None
            and self.spectrum is not None
            and self.step is not None and self.order is not None):
            if len(self.fit_lines) > 0:
                fit_axis = self._get_axis()
                
                # remove lines that are not in the spectral range
                spectrum_range = self._get_spectrum_range()

                if self.line_model == 'auto':
                    if self.apod == 1.0: fmodel = 'sinc'
                    else: fmodel = 'gaussian'
                else:
                    fmodel = str(self.line_model)
                    
                # guess fwhm
                fwhm_guess = utils.spectrum.compute_line_fwhm(
                    self.spectrum.shape[0]/1.25, self.step, self.order,
                    apod_coeff=self.apod,
                    wavenumber=self.wavenumber)
                
                fit_results =  fit.fit_lines_in_spectrum(
                    self.spectrum,
                    self.get_fit_lines(),
                    self.step, self.order,
                    1, self.axis_corr,
                    fwhm_guess=fwhm_guess,
                    signal_range=[np.nanmin(spectrum_range),
                                  np.nanmax(spectrum_range)],
                    wavenumber=self.wavenumber,
                    fmodel=fmodel)

                if 'lines-params-err' in fit_results:
                    self.fitted_vector = np.copy(fit_results['fitted-vector'])
                    self._update_plot()
                    
                    # print results
                    par = fit_results['lines-params']
                    par_err = fit_results['lines-params-err']
                    vel = fit_results['velocity']
                    vel_err = fit_results['velocity-err']
                    
                    self.fit_results_store.clear()

                    def format_store(p, v, perr):
                        _store = np.zeros(7, dtype=float)
                        _store[2] = p[iline][0]
                        _store[3] = p[iline][1]
                        _store[4] = v[iline]
                        _store[5] = p[iline][3]
                        
                        _store = ['{:.2e}'.format(_store[i])
                                 for i in range(_store.shape[0])]

                        if self.wavenumber:
                            _store[0] = Lines().get_line_name(
                                utils.spectrum.cm12nm(
                                    self.fit_lines[iline]))
                        else:
                            _store[0] = Lines().get_line_name(
                                self.fit_lines[iline])

                        _store[1] = '{:.3f}'.format(self.fit_lines[iline])
                        _store[6] = '{:.1f}'.format(
                            p[iline][1] / perr[iline][1])
                        return _store

                        
                    for iline in range(len(self.fit_lines)):
                        store = format_store(par, vel, par_err)
                        store_err = format_store(par_err, vel_err, par_err)
                        store_err[0] = ''
                        store_err[1] = ''
                        store_err[6] = ''
                        self.fit_results_store.append(store)
                        self.fit_results_store.append(store_err)

                    
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
                return utils.spectrum.create_nm_axis(
                    self.spectrum.shape[0], self.step, self.order, self.axis_corr)
            else:
                return utils.spectrum.create_cm1_axis(
                    self.spectrum.shape[0], self.step, self.order, self.axis_corr)
        else: return None
            
    def _get_spectrum_range(self):
        """Return min and max wavelength of the spectrum"""
        axis = self._get_axis()
        if axis is not None:
            if self.fitrange is not None:
                return self.fitrange[0], self.fitrange[1]
                
            elif self.spectrum is not None:
                nonans = np.nonzero(~np.isnan(self.spectrum))
                return np.nanmin(axis[nonans]), np.nanmax(axis[nonans])
            else:
                return None
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
                nm_max, nm_min = utils.spectrum.cm12nm(
                    self._get_spectrum_range())
            else:
                nm_min, nm_max = self._get_spectrum_range()
                
        if line_name == 'Sky lines':
            delta_nm = utils.spectrum.compute_line_fwhm(
                self.spectrum.shape[0], self.step, self.order,
                apod_coeff=self.apod, wavenumber=False)
                
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
            new_lines = utils.spectrum.nm2cm1(new_lines)

        
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
                    utils.spectrum.cm12nm(line))
            else:
                store[0] = Lines().get_line_name(line)
                        
                store[1] = '{:.3f}'.format(line)
            self.fit_results_store.append(store)        

        self._update_plot()
        
    def _set_line_name_cb(self, w):
        """Callback to set the line names

        :param w: Widget.
        """
        self.line_name = self._get_lines_keys()[w.get_active()]

    def _set_line_model_cb(self, w):
        """Callback to set the line model

        :param w: Widget.
        """
        self.line_model = self.line_models[w.get_active()]


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
            self.step = float(step)
            if self.parent is not None:
                self.parent.step = float(step)
                self._update_plot()

    def set_order(self, order):
        """Set order.

        :param order: Order.
        """
        if order is not None:
            self.worder.set_text('{}'.format(order))
            self.order = float(order)
            if self.parent is not None:
                self.parent.order = float(order)
                self._update_plot()
            
    def set_apod(self, apod):
        """Set apodization.

        :param apod: Apodization function.
        """
        if apod is None: apod = 1.0
            
        self.wapod.set_text('{}'.format(apod))
        self.apod = apod
        if self.parent is not None:
            self.parent.apod = float(apod)
            
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
            if np.size(self.spectrum) == np.size(spectrum):
                if np.any(self.spectrum[nonans] != spectrum[nonans]):
                    update = True
            else: update = True
                
        if update:
            self.spectrum = spectrum
            self.fitted_vector = None

    def set_fitrange(self, fitmin, fitmax):
        self.fitrange = (fitmin, fitmax)
        
    def get_frame(self):
        """Return the visual gtk.Frame object corresponding to the
        fitting plugin.
        """
        return self.fitframe

    def get_fit_lines(self):
        """Return the selected fit lines."""
        if self.fit_lines is not None:
            fit_lines = (np.array(self.fit_lines)
                         + np.array(utils.spectrum.line_shift(
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
    #shapes = ['rectangle', 'circle']
    shapes = ['circle']
    shape = None
    change_region_properties_cb = None
    regions = list()
    
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
        
        self.method = 'mean'
        menuitems[self.get_method_index('mean')].set_active(True)

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
        menuitems[self.get_shape_index(self.shape)].set_active(True)

        self.regions = list()
        
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
            self.change_region_properties_cb()

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
    

    def add_region(self, x, y):
        self.regions.append(Region(x, y, 10, self.get_shape()))

    def clear_regions(self):
        self.regions = list()
        
###########################################
### CLASS REGION ##########################
###########################################
class Region(object):

    shape = None
    x_range = None
    y_range = None
    selected_pixels = None
    
    def __init__(self, shape, xy_start, xy_stop, dimx, dimy):
        self.shape = shape
        xy_start = np.array(xy_start)
        xy_stop = np.array(xy_stop)
        
        if shape == 'circle':
            center = np.array((xy_start + xy_stop)/2.)
            radius = math.sqrt(np.sum((np.array(xy_stop) - center)**2.))
            X, Y = np.mgrid[0:dimx, 0:dimy]
            R = np.sqrt((X - center[0])**2. + (Y - center[1])**2.)
            self.selected_pixels = np.nonzero(R <= radius)
            
            x_range = np.array([np.nanmin(self.selected_pixels[0]),
                                np.nanmax(self.selected_pixels[0]) + 1])
            y_range = np.array([np.nanmin(self.selected_pixels[1]),
                                np.nanmax(self.selected_pixels[1]) + 1])

        else:
            
            y_range = np.array([min(xy_start[0], xy_stop[0]),
                                max(xy_start[0], xy_stop[0])])
            x_range = np.array([min(xy_start[1], xy_stop[1]),
                                max(xy_start[1], xy_stop[1])])

            
        # check range
        x_range[np.nonzero(x_range < 0)] = 0
        x_range[np.nonzero(x_range > dimx)] = dimx
        y_range[np.nonzero(y_range < 0)] = 0
        y_range[np.nonzero(y_range > dimy)] = dimy

        if x_range[1] - x_range[0] < 1.:
            if x_range[0] < dimx:
                x_range[1] = x_range[0] + 1
            else:
                x_range = [dimx - 1, dimx]

        if y_range[1] - y_range[0] < 1.:
            if y_range[0] < dimy:
                y_range[1] = y_range[0] + 1
            else:
                y_range = [dimy - 1, dimy]

        if shape == 'rectangle':
            X, Y = np.mgrid[x_range[0]:x_range[1],
                            y_range[0]:y_range[1]]
            self.selected_pixels = (
                X.flatten().astype(int),
                Y.flatten().astype(int))

        self.x_range = x_range
        self.y_range = y_range

    def get_selected_pixels(self):
        return self.selected_pixels

    



###########################################
### CLASS IMAGECANVAS #####################
###########################################
class ImageCanvas(Tools):

    
    properties = {'autocut' : True,
                  'autocenter' : True,
                  'autozoom' : True}

    autocuts = None # an Autocuts instance
    _autocut_method_changed = True

    cmaps = ['gray', 'gist_rainbow', 'jet', 'cool', 'hot', 'autumn', 'summer', 'winter']
    cmap = None
    cmap_contrast = 1.
    cmap_center = 0.5
    cbar = None

    regions = None # a Regions instance
    artists = list()
    
    vlim = None, None # vmin, vmax
    image = None # a 2d numpy array
    im = None # an AxesImage matplolib instance returned by imshow()
    
    wcs = None

    _init = True
    _xystart = None
    
    mode = None

    _last_xlim = None
    _last_ylim = None
    _last_vlim = None
    _last_arr8 = None
    _last_arr8_map = None
    _last_event = None

    signals = dict()
    region_selector = None

    scatter_params = None
    
    def __init__(self, regions, **kwargs):

        Tools.__init__(self, **kwargs)
        
        SIZE = (8,6)
        DPI = 75
        self._init = True
        self.fig = Figure(
            figsize=SIZE, dpi=DPI, tight_layout=False)
        self.subplot = self.fig.add_subplot(111, clip_on=False)
        self.imagebox = gtk.VBox()
        self.canvas = FigureCanvas(self.fig)
        self.imagebox.pack_start(self.canvas, fill=True, expand=True)

        # load regions instance
        self.regions = regions
        if self.regions is not None:
            self.regions.change_region_properties_cb = self._change_region_properties_cb
            # selector
            self._create_region_selector()


        self.autocuts = AutoCuts()
        self._autocut_method_changed = True
        self.cmap = 'gray'
        
        # events
        self.connect_all()


    def connect(self, event_name, callback_function):
        self.signals[event_name] = callback_function

    def _create_region_selector(self):
        rectprops = dict(facecolor='red', edgecolor='red',
                         alpha=0.8, fill=False, linewidth=2)
        if self.region_selector is not None:
            self.region_selector.to_draw.set_visible(False)
            self.region_selector.canvas.draw()
        if self.regions.get_shape() == 'circle':
            self.region_selector = EllipseSelector(
                self.subplot, self._on_select_region_cb,
                button=1, rectprops=rectprops, interactive=True)
        else:
            self.region_selector = RectangleSelector(
                self.subplot, self._on_select_region_cb,
                button=1, rectprops=rectprops, interactive=True)
        self.region_selector.state.add('square')
        
            
    def connect_all(self):
        self.canvas.mpl_connect('scroll_event', self._zoom_cb)
        self.canvas.mpl_connect('button_press_event', self._button_press_cb)
        self.canvas.mpl_connect('button_release_event', self._button_release_cb)
        self.canvas.mpl_connect('motion_notify_event', self._on_move_cb)
        self.canvas.mpl_connect('key_press_event', self._key_press_cb)
        self.canvas.mpl_connect('key_release_event', self._key_release_cb)
        self.canvas.mpl_connect('figure_enter_event', self._figure_enter_cb)
        self.grab_focus()

    def grab_focus(self):
        self.canvas.grab_focus()


    def _figure_enter_cb(self, event):
        self.grab_focus()

    def _change_region_properties_cb(self):
        self._create_region_selector()
        
    def _key_press_cb(self, event):
        if 'control' in event.key or 'ctrl' in event.key:
            self.mode = 'ctrl'
        elif 'alt' in event.key:
            self.mode = 'alt'

    def _key_release_cb(self, event):
        self.mode = None

    def _is_double_event(self, event):
        is_double = False
        if self._last_event is not None:
            if (event.x == self._last_event.x
                and event.y == self._last_event.y
                and event.button == self._last_event.button):
                is_double = True
        
        self._last_event = event
        return is_double
       
    
    def _button_press_cb(self, event):
        pass
    
    def _button_release_cb(self, event):
        self._xystart = None
        self._xystartdata = None

    def _on_select_region_cb(self, eclick, erelease):
        xy_start = (eclick.xdata, eclick.ydata)
        xy_stop = (erelease.xdata + 1, erelease.ydata + 1)
        
        region = Region(self.regions.get_shape(),
                        xy_start, xy_stop,
                        self.image.shape[0],
                        self.image.shape[1])

        if 'on-select-region' in self.signals:
            self.signals['on-select-region'](region)

    def _on_move_cb(self, event):
        if self._is_double_event(event):
            return
        
        if self._xystart is not None:
            dragx = event.x - self._xystart[0]
            dragy = event.y - self._xystart[1]
            scaley = self.fig.get_figheight() * self.fig.get_dpi()
            scalex = self.fig.get_figwidth() * self.fig.get_dpi()
            if event.button == 3:

                if self.mode is None:
                    # contrast
                    if dragy != 0:
                        self.cmap_contrast *= 1. + 0.2 * (dragy/abs(dragy)) 
                
                    # center
                    self.cmap_center = event.x / scalex
                    
                    if self.cmap_center > 1. : self.cmap_center = 1.
                    if self.cmap_center < 0. : self.cmap_center = 0.
                    
                if self.mode == 'ctrl': # image dragging
                    xlim = np.array(self.subplot.get_xlim())
                    ylim = np.array(self.subplot.get_ylim())
                    bbox = (self.subplot.get_xlim()[1]
                            - self.subplot.get_xlim()[0],
                            self.subplot.get_ylim()[1]
                            - self.subplot.get_ylim()[0])
                    xlim -= dragx * bbox[0] / scalex
                    ylim -= dragy * bbox[1] / scaley
                    
                    self.subplot.set_xlim(xlim)
                    self.subplot.set_ylim(ylim)
                    
                self.redraw()

            if event.button == 1:
                pass
                    
        self._xystart = event.x, event.y
        self._xydatastart = event.xdata, event.ydata
        
        if 'on-move' in self.signals:
            self.signals['on-move'](event.xdata, event.ydata)
       
    def _zoom_cb(self, event):
        xlim = self.subplot.get_xlim()
        ylim = self.subplot.get_ylim()
        xsize = xlim[1] - xlim[0]
        ysize = ylim[1] - ylim[0]
        xcenter = (xlim[1] + xlim[0]) / 2.
        ycenter = (ylim[1] + ylim[0]) / 2.
        if event.button == 'up':
            xsize *= 1.1
            ysize *= 1.1
        else:
            xsize *= 0.9
            ysize *= 0.9
        
        new_xlim = (xcenter - xsize/2., xcenter + xsize/2.)
        new_ylim = (ycenter - ysize/2., ycenter + ysize/2.)
        
        self.subplot.set_xlim(new_xlim)
        self.subplot.set_ylim(new_ylim)
        self.redraw()
            

    def get_widget(self):
        return self.imagebox

    def get_autocut_methods(self):
        return self.autocuts.get_autocut_methods()

    def set_autocut_method(self, autocut_method):
        self.autocuts.set_autocut_method(autocut_method)
        self._autocut_method_changed = True
        self.redraw()
            
    def get_cmaps(self):
        return self.cmaps

    def get_cmap(self):
        return self.cmap

    def set_cmap(self, cmap):
        if cmap in self.cmaps:
            self.cmap = cmap
        else:
            self._print_warning('Unknown colormap ({}), must be in {}'.format(
                cmap, self.cmaps))
        self.redraw()

    def _get_vlim(self):
        
        if self.vlim[0] is None or self._autocut_method_changed:

            if self.properties['autocut']:
                
                if np.all(np.isnan(self.image)):
                    self.vlim_cut = None, None

                else:
                    self.vlim_cut = self.autocuts.get_vlim(self.image)
                    self._autocut_method_changed = False

                
        if self.vlim_cut[0] is not None:
            vmin, vmax = self.vlim_cut
            
            vl = vmax - vmin
            vc = vl * self.cmap_center
            vc += vmin
            vl = vmax - vmin
            vl *= self.cmap_contrast
            vlim = [vc - vl/2., vc + vl/2.]
            self.vlim = min(vlim), max(vlim)

        return self.vlim
    

    def set_image(self, image, new=True):
        self.image = image.astype(float)
        self.vlim = None, None
        self.draw()

    def set_wcs(self, wcs):
        self.wcs = wcs

    def _get_arr8(self, recompute_all=True):
        
        cmap = self.cbar.get_cmap()
        vmin, vmax = self._get_vlim()

        if self._last_arr8_map is None:
            self._last_arr8_map = np.zeros_like(
                self.image, dtype=np.uint8)

        if recompute_all:
            self._last_arr8_map.fill(0)

        xmin, xmax = np.array(self.subplot.get_xlim()).astype(int)
        ymin, ymax = np.array(self.subplot.get_ylim()).astype(int)
        xmax += 1
        ymax += 1
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > self.image.shape[0]: xmax = self.image.shape[0]
        if ymax > self.image.shape[1]: ymax = self.image.shape[1]

        is_valid = False
        if vmin is not None and vmax is not None:
            if not np.isnan(vmin) and not np.isnan(vmax):
                is_valid = True

        if is_valid:
            if not np.all(self._last_arr8_map[xmin:xmax, ymin:ymax]):
                self._last_arr8 = orb.cutils.im2rgba(
                    self.image, self.cbar, vmin, vmax,
                    xmin, xmax, ymin, ymax,
                    self._last_arr8_map,
                    last_arr8=self._last_arr8,
                    res=1000)
                
                self._last_arr8_map[xmin:xmax, ymin:ymax] = 1
                has_changed = True
            
            else:
                has_changed = False
        else:
            self._last_arr8 = np.ones(
                (self.image.shape[1], self.image.shape[0], 4),
                dtype=np.uint8) * 255
            self._last_arr8_map.fill(1)
            has_changed = True
            
        self._last_vlim = (vmin, vmax)
        
        #arr8 = self.cbar.to_rgba(self.image_nonan.T, alpha=None, bytes=True)
        
        return self._last_arr8, has_changed

    def draw(self):
        vmin, vmax = self._get_vlim()
        self._last_cmap = str(self.cmap)

        if self._init:
            self.subplot.set_frame_on(True)
            self.subplot.get_xaxis().set_visible(False)
            self.subplot.get_yaxis().set_visible(False)

            # add colorbar
            if self.cbar is None:
                cax = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
                self.cbar = ColorbarBase(
                    cax, cmap=pl.get_cmap(self.cmap),
                    spacing='proportional')

            if self.cbar.get_clim() != (vmin, vmax):
                self.cbar.set_clim([vmin, vmax])
                self.cbar.draw_all()

            self.im = self.subplot.imshow(
                self._get_arr8(recompute_all=True)[0], origin='lower', interpolation='nearest',
                vmin=vmin, vmax=vmax, cmap=pl.get_cmap(self.cmap),
                aspect='equal')
            
            self.im.set_resample = True
        
            
        self._init = False
        self.redraw(recompute_all=True)
        
        
    def redraw(self, recompute_all=False):
        vlim_changed = False
        vmin, vmax = self._get_vlim()
        if (vmin, vmax) != self.im.get_clim():
            vlim_changed = True

        xylim_changed = False
        if self._last_xlim is None: xylim_changed = True
        elif self._last_ylim is None: xylim_changed = True
        elif self._last_xlim != self.subplot.get_xlim(): xylim_changed = True
        elif self._last_ylim != self.subplot.get_ylim(): xylim_changed = True
        
        cmap_changed = False
        if self.cmap != self._last_cmap:
            cmap_changed = True

        if xylim_changed or vlim_changed or cmap_changed or recompute_all:
            if vlim_changed or cmap_changed:
                if vlim_changed:
                    self.im.set_clim([vmin, vmax])
                    self.cbar.set_clim([vmin, vmax])
           
                if cmap_changed:
                    cmap = pl.get_cmap(self.cmap)
                    self.im.set_cmap(cmap)
                    self.cbar.set_cmap(cmap)
                    
                self.cbar.draw_all()
                recompute_all = True

            arr8, has_changed = self._get_arr8(
                recompute_all=recompute_all)

            if has_changed:
                # break through : replace self.im.set_data(arr8). Much
                # much faster when input data has no nans or infs
                self.im._A = arr8

            self.im.changed() # very important, avoid 'zooming' bug
            
            if self.scatter_params is not None:
                self.subplot.scatter(*self.scatter_params)
                
            self.canvas.draw()
            
        self._last_xlim = self.subplot.get_xlim()
        self._last_ylim = self.subplot.get_ylim()
        self._last_cmap = str(self.cmap)
        

    def scatter(self, *args, **kwargs):
        self.scatter_params = args
        self.redraw(recompute_all=True)
    





###########################################
### CLASS IMAGEWINDOW #####################
###########################################
class ImageWindow(PopupWindow):
    """Implement a window for plotting zaxis data."""

    def __init__(self, image):
        """Init and construct spectrum window.
        """
        SIZE = (8,6)
        DPI = 75
        PopupWindow.__init__(self, title='Image',
                             size=(SIZE[0]*DPI,SIZE[1]*DPI))
   
        self.fig = Figure(figsize=SIZE, dpi=DPI)
        self.subplot = self.fig.add_subplot(111)


        # CREATE framebox
        framebox = gtk.VBox()
        # add figure canvas
        self.canvas = ImageCanvas(None)  # a gtk.DrawingArea
        self.canvas.set_image(image)
        framebox.pack_start(self.canvas.get_widget(),
                            fill=True, expand=True)

        # add framebox
        self.w.add(framebox)


    def scatter(self, *args, **kwargs):
        self.canvas.scatter(*args, **kwargs)
