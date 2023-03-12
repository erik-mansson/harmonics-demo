#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An illustration of angular distributions (spherical harmonics, beta-parameters or Legendre polynomials).

Created 5 February 2023

@author: erik.maansson@desy.de
"""

import os, sys, time, traceback, enum
from PyQt5 import QtWidgets, QtCore
from PyQt5 import uic
from PyQt5.QtCore import Qt, QLocale, pyqtSlot
from PyQt5.QtGui import QTransform
import numpy as np
import scipy.special
import pyqtgraph as pg
# import pyabel  # Abel transforms
# import shtools  # Spherical harmonics

try:
    import numba
    @numba.vectorize([numba.float32(numba.complex64),
                      numba.float64(numba.complex128)])
    def abs2(array):
        """Compute np.abs(array)**2 in a more efficient way."""
        return array.real**2 + array.imag**2
except ModuleNotFoundError:
    def abs2(array):
        """Compute np.abs(array)**2 without np.sqrt(), this version without numba just needs more memory."""
        return array.real**2 + array.imag**2


class ComplexMode(enum.Enum):
    """How and when to convert complex numbers to real for display.
    
    If the computation has multiple terms (each being a spherical harmonic)
    the first four modes sum the terms coherently, i.e. as complex numbers,
    while the incoherent mode sums after converting to absolute value squared.
    
    Either way, some conversion to real numbers happens before the ProjectionMode's
    summing (integrating) over flattened coordinates or radius is applied.
    """
    abs = 'Absolute value |...|'
    square = 'Intensity |...|²'
    real = 'Real value Re(...)'
    imag = 'Imaginary value Im(...)'
    incoherent = 'Incoherent sum |.|²+|.|²+|.|²'  # Incoherent squared sum of terms


class ProjectionMode(enum.Enum):
    sum = '(x,y) 2D-projection of all'
    slice = '(x,y) 2D-slice (central 1%)'
    spherical = '(phi,theta) Spherical projection'


class Width(enum.Enum):
    Narrow = 0.02
    Wide = 0.08


pg.setConfigOptions(imageAxisOrder='row-major')

#def legendre(l, m, z):
#    """Evaluate the associated Legendre polynomial (l, m) at z."""
##    polynomial = 0.0
##    for k in range(m, l + 1):
##        coefficient = (big expression with factorials and combinatoric "N over K")
##        polynomial += np.pow(z, k - m)
##    return ((-1)**m * 2**l * np.pow(1 - z**2, m/2)) * polynomial
#    # Ah, scipy.special already has it
#    return scipy.special.lpmv(m, l, self.cos_theta)    


class AngularGUI(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        """Construct the window."""
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # a good way of making sure also timers stop when closing application
        self.raise_()  # bring window to front, useful when starting in Spyder
        
        # To include icons. (Compiled using "rcc -binary resources.qrc -o resources.rcc" on Linux)
        try:
            folder = os.path.dirname(__file__)
        except NameError:  # Running in interactive console so no script __file__ defined
            folder = '.'
        uic.loadUi(os.path.join(folder, 'angular_distribution_demo.ui'), self)
        
        
        # Configuration (constant while running)
        self.count = 3  # The number of terms (rings) in the GUI
        self.dtype = np.float64
        # self.dtype = np.float32  # Save memory, accepting more rounding errors. On 64bit processor it might however even be slower.
        
        # Instance variables
        self.complexMode = ComplexMode.abs
        self.projectionMode = ProjectionMode.sum
        self.ring_radii = np.ones(self.count, dtype=self.dtype)  # [px] allowed range is 0 to self.pixels // 2
        self.ring_widths = np.zeros(self.count, dtype=self.dtype)  # [px] allowed range is 0 to self.pixels // 10
        self.ls = np.arange(0, self.count, dtype=np.int16) + 1  # l-number of Y_l^m
        self.ms = np.arange(0, self.count, dtype=np.int16) + 1  # m-number of Y_l^m
        self.amplitudes = np.zeros(self.count, dtype=self.dtype)  # allowed range is 0 to np.pi * self.pixels**2
        self.phases = np.zeros(self.count, dtype=self.dtype)  # [rad] the GUI's range is 0 to 2*pi
        self.result = None  # 3D-array of complex numbers, defined in recompute(). Most diagrams are produced from this.
        self.summed_coherently = None
        self.pixels = 0  # Determines image size. Larger is slower.
        
        self.radius_widgets = []
        self.width_widgets = []
        self.l_widgets = []
        self.m_widgets = []
        self.amplitude_widgets = []
        self.phase_widgets = []
        # self.term_plots = []
        for i in range(self.count):
            self.radius_widgets.append(getattr(self, f'_r{i}'))
            self.width_widgets.append(getattr(self, f'_w{i}'))
            self.l_widgets.append(getattr(self, f'_l{i}'))
            self.m_widgets.append(getattr(self, f'_m{i}'))
            self.amplitude_widgets.append(getattr(self, f'_amplitude{i}'))
            self.phase_widgets.append(getattr(self, f'_phase{i}'))
            # self.term_plots.append(self._termsPlot.plot([0.0, 1.0], [1 + 2 * i, 1 + 2 * i]))
        
        # Connect widget signals to slots (methods) that handle them
        self._imageSize.valueChanged.connect(self.onImageSizeChange)
        for widget in [*self.radius_widgets, *self.width_widgets,
                       *self.l_widgets, *self.m_widgets,
                       *self.amplitude_widgets, *self.phase_widgets]:
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.recompute)
            else:
                widget.currentIndexChanged.connect(self.recompute)
        self._complexMode.currentIndexChanged.connect(self.onComplexModeChange)
        self._projectionMode.currentIndexChanged.connect(self.updateDiagrams)
        self._defaultBtn.clicked.connect(self.setDefaults)  # TODO
        self._randomBtn.clicked.connect(self.randomize)
        self._rephaseBtn.clicked.connect(self.randomizePhase)
        
        # Create diagrams
        self._imageLayout.ci.layout.setContentsMargins(0, 0, 0, 0)  # remove outer margins
        self._imageLayout.ci.setSpacing(0)  # remove margins between subdiagrms
        self.imageContainer = self._imageLayout.addPlot()
        self.imageContainer.setDefaultPadding(0.0)
        # self.imageContainer.hideAxis('left')
        # self.imageContainer.hideAxis('bottom')
        self.image = pg.ImageItem()
        self.imageContainer.addItem(self.image)
        # Contrast/colormap control
        # self.colorbar = pg.ColorBarItem(width=15, rounding=1E-6); self.colorbar.setImageItem(self.image)  # This is smaller, but didn't get it to update to actual numeric range of image intensities
        # self.colorbar = CustomizedHistogramLUTItem(image=self.image,,
        #     histogramWidth=(10, 30),  # reduce histogram width from default (45, 152)
        #     rectSize=9,  # reduce colorbar width from default 15
        #     tickSize=9)  # reduce triangular color marker width from default 15
        self.colorbar = pg.HistogramLUTItem(image=self.image)
        self.colorbar.gradient.rectSize = 9
        self.colorbar.gradient.tickSize = 9
        self.colorbar.gradient.setMaxDim(9 + 9)
        self.colorbar.layout.setContentsMargins(0, 0, 0, 0)  # reduce margins from default 1
        self.colorbar.vb.setMinimumWidth(10)
        self.colorbar.vb.setMaximumWidth(20)
        self.colorbar.vb.setLimits(minYRange=1E-12)  # When numerically just noise near zero, don't zoom all the way in
        self._imageLayout.addItem(self.colorbar)
        self.imageContainer.setMenuEnabled(False, None)  # hide the non-useful 'Plot Options' part of context menu for image
        self._imageLayout.setBackground('#111')
        self._verticalPlot.setBackground('#111')
        self._radialPlot.setBackground('#111')
        self._angularPlot.setBackground('#111')
        
        self.vertical_projection = self._verticalPlot.plot([0.0, 1.0], [0.0, 0.0], pen='b')
        self._verticalPlot.setLabel('left', 'z (px)')
        self._verticalPlot.setLabel('bottom', 'Intensity (arb. u.)')
        self._verticalPlot.setLimits(minXRange=1E-12)  # When numerically just noise near zero, don't zoom all the way in
        self._verticalPlot.setDefaultPadding(0.0)
        self._verticalPlot.setYLink(self.imageContainer)
        
        self.radial_distribution = self._radialPlot.plot([0.0, 1.0], [0.0, 0.0], pen='y')
        self._radialPlot.setLabel('bottom', 'Radius (px)')
        self._radialPlot.setLimits(minYRange=1E-12)  # When numerically just noise near zero, don't zoom all the way in
        self._radialPlot.setDefaultPadding(0.0)
        
        self.angular_distribution = self._angularPlot.plot([0.0, 1.0], [0.0, 0.0], pen='y')
        self._angularPlot.setLimits(minYRange=1E-12)  # When numerically just noise near zero, don't zoom all the way in
        self._angularPlot.setDefaultPadding(0.01)
        # self._angularPlot.hideAxis('left')
        
        self.onImageSizeChange()  # Updates self.pixels and calls self._define_coordinates() and self.recompute()
        

    def _define_coordinates(self):
        self.setCursor(Qt.WaitCursor)
        # Cached coordinate arrays
        self.abscissa = np.arange(self.pixels, dtype=self.dtype) - self.pixels // 2
        # The VMI-flattened coordinate (y) is first for simplicity
        y = self.abscissa[:, None, None]  # varies along axis 0
        self.z = self.abscissa[None, :, None]  # varies along axis 1
        x = self.abscissa[None, None, :]  # varies along axis 2
        self.xy_axes_indices = (0, 2)
        
        r_xy = np.hypot(x, y)
        self.phi = np.arctan2(y, x)  # -pi to +pi radians
        self.radius = np.hypot(r_xy, self.z)
        self.phi_broadcasted = np.broadcast_to(self.phi, self.radius.shape)  # full 3D array
        self.theta = np.arctan2(r_xy, self.z)  # 0 to pi radians
        # self.theta = np.arccos(self.z / self.radius); self.theta[self.radius == 0] = 0  # OK alternative
        # assert np.max(np.abs(self.z - self.radius * np.cos(self.theta))) < 1E-10
        assert np.max(np.abs(self.z - self.radius * np.cos(self.theta))) < 1E-4
        
        # Currently not needed:
        # with np.errstate(divide='ignore', invalid='ignore'):  # Don't print warnings about division by zero here
        #     self.cos_theta = self.z / self.radius  # Equals np.cos(self.theta) except the NaN at self.radius==0
        # self.cos_theta[np.isnan(self.cos_theta)] = 1.0  # To agree with what np.arctan2(0.0, 0.0) gives
        # assert np.max(np.abs(np.cos(self.theta) - self.cos_theta)) < 1E-10
        
        self.unsetCursor()

    def updateArrays(self):
        """Convert from widgets to instance variable arrays."""
        for i in range(self.count):
            self.ls[i] = self.l_widgets[i].value()
            self.ms[i] = self.m_widgets[i].value()
            if self.ms[i] > self.ls[i]:
                self.ms[i] = self.ls[i]
                self.m_widgets[i].setValue(self.ms[i])
            elif self.ms[i] < -self.ls[i]:
                self.ms[i] = -self.ls[i]
                self.m_widgets[i].setValue(self.ms[i])
            # self.ring_radii[i] = self.radius_widgets[i].value() * 0.01 * (self.pixels // 2) # [px]
            self.ring_radii[i] = self.radius_widgets[i].value() * (0.009) * (self.pixels // 2) # [px] with 11.11% margin (so wide rings don't go much outside grid)
            self.ring_widths[i] = Width[self.width_widgets[i].currentText()].value * (self.pixels // 2) # [px]
            self.amplitudes[i] = np.pi * self.pixels**2 * self.amplitude_widgets[i].value() / self.amplitude_widgets[i].maximum()
            self.phases[i] = 2 * np.pi * np.mod(self.phase_widgets[i].value() / self.phase_widgets[i].maximum() - 0.25, 1.0)

    def onImageSizeChange(self):
        pixels = int(self._imageSize.value())
        if pixels > 401:  # Avoid really slow, large images
            pixels = (min(pixels, 401) // 2) * 2 + 1
            self._imageSize.setValue(pixels)
        if pixels % 2 != 1:  # Ensure that an odd integer
            pixels = (pixels // 2) * 2 + 1
            self._imageSize.setValue(pixels)
        if pixels != self.pixels:
            self.pixels = pixels
            self._define_coordinates()
            self.recompute()

    def onComplexModeChange(self):
        """Do either a full recomputation or just a projection update."""
        sum_coherently = ComplexMode(self._complexMode.currentText()) != ComplexMode.incoherent
        if sum_coherently is self.summed_coherently:
            self.updateDiagrams()  # Just update the shown projection
        else:
            self.recompute()  # Redo entire calculation

    @pyqtSlot()
    def recompute(self):
        """ Recompute self.result, and call updateDiagrams()."""
        self.setCursor(Qt.WaitCursor)
        self._progress.setMaximum(np.sum(self.amplitudes != 0) * 4 + 2)
        self._progress.setValue(1)
        self.updateArrays()
        sum_coherently = ComplexMode(self._complexMode.currentText()) != ComplexMode.incoherent
        t0 = time.time()
        result = 0.0
        for i in range(self.count):
            if self.amplitudes[i] == 0 and np.any(self.amplitudes[i + 1:self.count] != 0):
                self._progress.setValue(self._progress.value() + 4)
                continue  # Save time by not adding arrays of zeros
            # Radial dependence (Gaussian)
            term = np.power(2.0, -((self.radius - self.ring_radii[i]) / self.ring_widths[i])**2, dtype=self.dtype)
            # Normalize (numerically should be better than analytic here, since radial width is finite)
            term *= (self.dtype(self.amplitudes[i]) / term.sum(dtype=self.dtype))
            self._progress.setValue(self._progress.value() + 1)
            if self.amplitudes[i] != 0:
                # Angular dependence TODO np.complex64 2D-array spherical harmonic
                # term *= legendre(self.ls[i], self.ms[i], self.cos_theta)
                # if self.ms[i] != 0:
                #     term *= np.exp(1j * (self.ms[i] * self.phi + self.phases[i]))
                # elif self.phases[i] != 0.0:
                #     term *= np.exp(1j * self.phases[i])
                # Nah, simply use scipy's implementation! It also includes the the normalization
                # sqrt((2l+1) * (l-m)! / (n+m)! / 4 / pi), matching what Wikipedia says for quantum mechanical application.
                if self.dtype is np.float32:
                    term = term * scipy.special.sph_harm(self.ms[i], self.ls[i], self.phi, self.theta, dtype=np.complex64)
                else:
                    term = term * scipy.special.sph_harm(self.ms[i], self.ls[i], self.phi, self.theta)
                self._progress.setValue(self._progress.value() + 2)
                if self.phases[i] != 0.0:
                    if self.dtype is np.float32:
                        term *= np.complex64(np.exp(1j * self.phases[i]))
                    else:
                        term *= np.exp(1j * self.phases[i])
            else:
                self._progress.setValue(self._progress.value() + 2)
            if sum_coherently:
                result = result + term
            else:  # 'Incoherent sum of squared abs. |term|²+|term|²+|term|²+...'
                result = result + abs2(term)
            self._progress.setValue(self._progress.value() + 1)
        self.result = result
        self.summed_coherently = sum_coherently
        self.unsetCursor()
        self.updateDiagrams()
        duration = time.time() - t0
        if duration > 0.8:
            message = f"Recomputed in {duration:.2f} s for {self.pixels} px {self.result.dtype}."
            self.statusbar.showMessage(message, 3000)
            print(message)
        else:
            self.statusbar.showMessage('', 1)

    def updateDiagrams(self):
        complex_mode = ComplexMode(self._complexMode.currentText())
        if complex_mode == ComplexMode.abs:
            data = np.abs(self.result)
            self._verticalPlot.setLabel('bottom', 'Absolute value (arb. u.)')
        elif complex_mode == ComplexMode.square:
            data = abs2(self.result)
            self._verticalPlot.setLabel('bottom', 'Intensity (arb. u.)')
        elif complex_mode == ComplexMode.real:
            data = self.result.real
            self._verticalPlot.setLabel('bottom', 'Real value (arb. u.)')
        elif complex_mode == ComplexMode.imag:
            data = self.result.imag
            self._verticalPlot.setLabel('bottom', 'Imaginary value (arb. u.)')
        else:  # ComplexMode.incoherent needs no conversion
            data = self.result
            self._verticalPlot.setLabel('bottom', 'Intensity sum (arb. u.)')
        assert np.issubdtype(data.dtype, np.floating), 'Data should be real (not complex) numbers'
        self._progress.setValue(self._progress.maximum() - 1)
        
        projection = ProjectionMode(self._projectionMode.currentText())
        # HINT: The flattening Abel projection along y (or radius) is always considered as incoherent,
        # i.e. no quantum interference between particles reaching MCP at different times (or energies).
        if projection == ProjectionMode.spherical:
            # 2D-angular distribution (radially integrated).
            # If there were only 2D data of angles (i.e. one term of spherical harmonic),
            # that 2D grid would be directly usable for as image coordinates. But when
            # also a range of radii within the 3D-array, one should sum over all radii.
            # True integration would need the integration volume element (Jacobian) as weight,
            # is it still neded for this "nearest point" interpolation approach at fixed (theat, phi) grid?
            # Yes, I think one still would want to multiply e.g. by radius, so that if
            # the value is -1 at a small radius and +1 at a large radius (where more volume 
            # within theta,phi grid) one would want the integral to be positive rather than zero.
            # interpolator = scipy.interpolate.griddata(points_of_values, values, points_to_interpolate_at,
            #                                           'nearest', fill_value=0) # or 'linear', fill_value=0)
            # A good and efficient alternative would be numerical bin assignments like in Erik's VMI 2D PolarTransform,
            # then the number of points assigned to same bin already achieves the desired radial dependent effect.
            # A simple alternative is to make numpy.histogram2d do the calculation without preassigned bins:
            theta_edges = np.linspace(0, np.pi, int(180 * np.sqrt(self.pixels) / 50))
            phi_edges = np.linspace(-np.pi, np.pi, int(360 * np.sqrt(self.pixels) / 75))
            result_2D, _, _ = np.histogram2d(self.theta.ravel(), self.phi_broadcasted.ravel(),
                                             [theta_edges, phi_edges], weights=data.ravel())
            result_2D = np.flipud(result_2D)  # flip the theta-axis, to get min theta (max z) on top
            
            # Special averaging or smoothing near phi=0
            middle_phi_region = result_2D[:, result_2D.shape[1] // 2 - 1:result_2D.shape[1] // 2 + 1]
            middle_phi_region[...] = middle_phi_region.mean(1, keepdims=True)
            # Smoothing along phi, wrapping around
            result_2D = (np.concatenate((result_2D[:, 1:], result_2D[:, :1]), 1)
                          + 2 * result_2D
                          + np.concatenate((result_2D[:, -1:], result_2D[:, :-1]), 1)) / 4
            # Smoothing along theta, non-wrapping
            result_2D = (np.concatenate((result_2D[1:], result_2D[-1:]), 0)
                         + 2 * result_2D
                         + np.concatenate((result_2D[:1], result_2D[:-1]), 0)) / 4
            
            self.imageContainer.setLabel('bottom', 'phi (rad)')
            self.imageContainer.setLabel('left', 'theat (rad)')
            self._verticalPlot.setYLink(None)
            self.image.getViewBox().setAspectLocked(False)
            self.image.setTransform(QTransform().scale(phi_edges[1] - phi_edges[0],
                                                       theta_edges[1] - theta_edges[0]
                                                       ).translate(-np.pi, 0))
        else:
            if projection == ProjectionMode.slice:
                # Select central z-slice that closest approximates 1% of array length (may be up to 2%).
                middle_index = data.shape[0] // 2
                index_halfrange = middle_index // 100
                result_2D = data[middle_index - index_halfrange:middle_index + index_halfrange + 1, :, :].sum(0)
            else:
                result_2D = data.sum(0)
            self.imageContainer.setLabel('bottom', 'x (px)')
            self.imageContainer.setLabel('left', 'z (px)')
            self.image.setTransform(QTransform().translate(-self.pixels / 2, -self.pixels / 2))
            self.image.getViewBox().setAspectLocked(True)
            self._verticalPlot.setYLink(self.imageContainer)
        self.image.setImage(result_2D, autoLevels=True)
        self.imageContainer.autoRange()
        
        self._progress.setValue(self._progress.maximum())
        
        # Vertical dependence (z-cordinate)
        self.vertical_projection.setData(data.sum(self.xy_axes_indices), self.z[0, :, 0])
        self._verticalPlot.enableAutoRange(x=True)
        
        # Radial dependence (angularly integrated or averaged):
        radius = self.abscissa[self.abscissa.shape[0] // 2:]  # Steps of 1 pixel
        # radius = np.linspace(0, self.abscissa[-1], self.pixels)  # Steps of 0.5 pixels
        bin_edges = np.concatenate((radius[:1] - 0.5, radius + 0.5))  # Halfway between the radii
        result_1D, _ = np.histogram(self.radius, bin_edges, weights=data)
        if projection == ProjectionMode.slice:
            # Reusing this setting to here normalize by bin count, i.e. average instead of sum
            count, _ = np.histogram(self.radius, bin_edges)
            if self.dtype is np.float32:
                count = self.dtype(count)
            result_1D = result_1D / count
            self._radialPlot.setLabel('left', 'Angle-averaged')
        else:
            self._radialPlot.setLabel('left', 'Angle-integrated')
        self.radial_distribution.setData(radius, result_1D)
        self._radialPlot.enableAutoRange()
        
        
        if projection == ProjectionMode.slice:
            # Reusing the ProjectionMode to here use theta instead of phi:
            # 1D theta-angular distribution (radially and phi-integrated)
            bin_edges = np.linspace(0, np.pi, int(180 * np.sqrt(self.pixels) / 50) // 2 * 2 + 1)
            # N = 26  # OK
            N = 36  # Heuristic tweaked choice of bins to reduce pattern "noise" (rounding artefacts)
            bin_edges = np.linspace(0, np.pi, N)
            result_1D, _ = np.histogram(self.theta.ravel(), bin_edges, weights=data.ravel())
            # Apply a symmetric smoothing kernel:
            result_1D = (np.concatenate((result_1D[1:], result_1D[-1:]))
                         + 2 * result_1D
                         + np.concatenate((result_1D[:1], result_1D[:-1]))) / 4
            self._angularPlot.setLabel('bottom', '')
            self._angularPlot.setLabel('left', 'theta (rad)')
            self._angularPlot.invertY(True)
            self.angular_distribution.setData(result_1D, (bin_edges[1:] + bin_edges[1:]) / 2)
        else:
            # 1D phi-angular distribution (radially and theta-integrated)
            # bin_edges = np.linspace(-np.pi, np.pi, int(360 * np.sqrt(self.pixels) / 50) // 2 * 2)
            N = 52  # Heuristic tweaked choice of bins to reduce pattern "noise" (rounding artefacts)
            bin_edges = (np.arange(N + 3) - N / 2 - 1) / N * 2 * np.pi
            result_1D, _ = np.histogram(self.phi_broadcasted.ravel(), bin_edges, weights=data.ravel())
            result_1D = np.concatenate((result_1D[1:2] + result_1D[-1:] / 2, result_1D[2:-2], result_1D[-2:-1] + result_1D[-1:] / 2))
            bin_edges = bin_edges[1:-1]
            # N=26 and N=52 are very uniform except for negative and positive deviations at fixed indices, which we can correct for:
            result_1D[25:27] = np.mean(result_1D[25:27])
            result_1D[12:14] = np.mean(result_1D[12:14])
            result_1D[38:40] = np.mean(result_1D[38:40])
            smoothed = (result_1D[::2] + result_1D[1::2]) / 2
            result_1D[::2] = (result_1D[::2] + smoothed) / 2
            result_1D[1::2] = (result_1D[1::2] + smoothed) / 2
            result_1D = (result_1D + np.concatenate((result_1D[1:], result_1D[:1]))) / 2
            self._angularPlot.setLabel('left', '')
            self._angularPlot.setLabel('bottom', 'phi (rad)')
            self._angularPlot.invertY(False)
            self.angular_distribution.setData((bin_edges[1:] + bin_edges[1:]) / 2, result_1D)
        self._angularPlot.autoRange()
        # Make y-range include 0
        if result_1D.min() > 0:
            self._angularPlot.setYRange(0, self._angularPlot.viewRange()[1][1])
        elif result_1D.max() < 0:
            self._angularPlot.setYRange(self._angularPlot.viewRange()[1][0], 0)
        
        self._progress.setValue(0)

    def randomize(self):
        for w in [*self.radius_widgets, *self.width_widgets,
                  *self.l_widgets, *self.m_widgets,
                  *self.amplitude_widgets, *self.phase_widgets]:
            w.blockSignals(True)  # don't trigger an update per iteration, call recompute() at end
        for widget in [*self.l_widgets, *self.phase_widgets]:
            widget.setValue(np.random.randint(widget.minimum(), widget.maximum() + 1))
        for widget, l_widget in zip(self.m_widgets, self.l_widgets):
            widget.setValue(np.random.randint(-l_widget.value(), l_widget.value() + 1))
        for widget in self.width_widgets:
            widget.setCurrentIndex(np.random.randint(0, widget.count()))
        # Define the radii to be in ascending order.
        for w, a in zip(self.radius_widgets,
                        sorted([np.random.randint(w.minimum(), w.maximum() + 1)
                                for w in self.radius_widgets])
                        ): w.setValue(a)
        # Choose amplitudes in ascending order, as large amplitude at small radius looks bad.
        for w, a in zip(self.amplitude_widgets,
                        sorted([np.random.randint(w.minimum(), w.maximum() + 1)
                                for w in self.amplitude_widgets])
                        ): w.setValue(a)
        for w in [*self.radius_widgets, *self.width_widgets,
                  *self.l_widgets, *self.m_widgets,
                  *self.amplitude_widgets, *self.phase_widgets]:
            w.blockSignals(False)
        self.recompute()

    def randomizePhase(self):
        for widget in self.phase_widgets:
            widget.blockSignals(True)  # don't trigger an update per iteration, call recompute() at end
            widget.setValue(np.random.randint(0, widget.maximum() + 1))
            widget.blockSignals(False)
        self.recompute()

    def setDefaults(self):
        for w in [*self.radius_widgets, *self.width_widgets,
                  *self.l_widgets, *self.m_widgets,
                  *self.amplitude_widgets, *self.phase_widgets]:
            w.blockSignals(True)  # don't trigger an update per iteration, call recompute() at end
        for w, a in zip(self.l_widgets, [0, 1, 0]): w.setValue(a)
        for w, a in zip(self.m_widgets, [0, 0, 0]): w.setValue(a)
        for w, a in zip(self.radius_widgets, [100, 75, 100]): w.setValue(a)
        for w, a in zip(self.amplitude_widgets, [0, 11, 20]): w.setValue(a)
        for w in self.width_widgets: w.setCurrentIndex(0)
        for w in self.phase_widgets: w.setValue(w.maximum() // 4)
        for w in [*self.radius_widgets, *self.width_widgets,
                  *self.l_widgets, *self.m_widgets,
                  *self.amplitude_widgets, *self.phase_widgets]:
            w.blockSignals(False)
        self.recompute()

    @pyqtSlot(str)
    def showError(self, message):
        """Show an error message in GUI statusbar.
        """
        file_and_line_number = '.'
        if len(sys.exc_info()) >= 3 and sys.exc_info()[-1]:  # there is info about file and line number
            dir = os.path.dirname(os.path.abspath(__file__))
            for tb in traceback.extract_tb(sys.exc_info()[-1]):
                if tb.filename.startswith(dir):  # while within modules of this program
                    file_and_line_number = '{} Line {} in {}.'.format('' if message.endswith('.') else '.',
                        tb.lineno, tb.filename[len(dir) + 1:])
                else:  # when we reach traceback parts within general libraries, don't show such a line number
                    break
        if not message.endswith(file_and_line_number):
            message += file_and_line_number
        print('ERROR: {}'.format(message))
        self.statusbar.showMessage(message, 5000)
        try:
            print(traceback.format_exc())  # print detailed error info (filename, line number, the Python code). 
        except Exception:
            pass
        
    # def closeEvent(self, event):
    #     super().closeEvent(event)


if __name__ == '__main__':  # To start the program when executed as a script (not just imported)
    def start_app():
        """Handle localization and catch otherwise uncaught errors."""
        locale = QLocale(QLocale.English, QLocale.Cyprus)  # decimal period, English, Monday
        options = QLocale.DefaultNumberOptions  # with thousands separators
        if hasattr(QLocale, 'OmitLeadingZeroInExponent'):  # only in newer Qt version
            options = options | QLocale.OmitLeadingZeroInExponent
        locale.setNumberOptions(options)
        QLocale.setDefault(locale)
        
        window = None
        last_keyboard_interrupt_timestamp = np.NaN
        
        def excepthook(exc_type, exc_value, exc_tb):
            nonlocal last_keyboard_interrupt_timestamp
            """Can be used as sys.excepthook to catch unhandled errors in Qt slots."""
            tb = " ".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            if window:
                if isinstance(exc_value, KeyboardInterrupt):
                    # To be able to terminate frozen program by Ctrl+C on command line!
                    now = time.time()
                    if now - last_keyboard_interrupt_timestamp <= 10.0:
                        # To not accidentally close when just trying to copy text 
                        # from console by mistaken Ctrl+C press, we only shut down
                        # if pressed twice within ten seconds.
                        print("*** Unhandled exception:\n", tb, flush=True)
                        window.close()
                    else:
                        print('WARNING: If you press Ctrl+C again within ten seconds, the program will be shut down.', flush=True)
                        last_keyboard_interrupt_timestamp = now
                else:
                    print("*** Unhandled exception:\n", tb, flush=True)
                    window.showError('{}: {}'.format(exc_type.__name__, exc_value))
            else:
                print("*** Unhandled exception:\n", tb, flush=True)
        
        # Not really needed within Spyder/IPython, they seem to catch and print anyway,
        # but may be good for running stand-alone and still getting any uncaught error printed.
        sys.excepthook = excepthook
        
        time.sleep(0.3)  # maybe a bit of waiting at start can reduce the risk of freezes?
        app = QtCore.QCoreApplication.instance()  # try to reuse in Spyder
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        else:
            print('Reused QApplication')
        QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
        window = AngularGUI()
        window.show()
        window.raise_()  # bring window to front, useful when starting in Spyder
        sys.exit(app.exec_())
    
    start_app()
