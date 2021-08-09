#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An illustration of harmonics (overtones) and the importance of their phase.

The main purpose is to illustrate how the phase of the harmonics need to be
controlled (synchronized) to make a train of short pulses in the sum-waeve.
This relates to the High-order Harmonic Generation process that can give
attosecond pulse trains when intense lasers are focused in a medium.

If the pyaudio package is installed, the sum waveformed can be heard too.

Created July 11 to 24, 2021

@author: erik.maansson@desy.de
"""

import os, sys, time, traceback
from PyQt5 import QtWidgets, QtCore
from PyQt5 import uic
from PyQt5.QtCore import Qt, QLocale, pyqtSlot
import numpy as np
import pyqtgraph as pg
try:
    import pyaudio
except ModuleNotFoundError:
    print('*** pyaudio not installed. Sound will not be available. ***')
    pyaudio = None


class HarmonicsGUI(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        """Construct the window."""
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # a good way of making sure also timers stop when closing application
        self.raise_()  # bring window to front, useful when starting in Spyder
        
        # To include icons. (Compiled using "rcc -binary resources.qrc -o resources.rcc" on Linux)
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'harmonics_demo.ui'), self)
        if pyaudio is None:
            self._volume.setEnabled(False)
            self._volume.setToolTip('Can not play as sound, because pyaudio is not installed.')
        
        # Configuration (constant while running)
        self.count = 8  # The number of harmonic orders handled by GUI
        self.amplitude_widget_factor = 20  # widget runs from 0 to 20, but we use it as 0.0 to 1.0
        self.periods_shown = 3
        self.sound_sampling_frequency = 44100.0  # [Hz]
        
        # Instance variables
        self.orders = np.arange(0, self.count, dtype=np.float64) + 1  # harmonic orders
        self.amplitudes = np.zeros(self.count, dtype=np.float64)  # allowed range is 0 to 1
        self.phases = np.zeros(self.count, dtype=np.float64)  # the GUI's range is 0 to 2*pi
        self.single_period = False
        self.time_axis = None  # [s] defined in soundSettingsChanged()
        self.result = None  # defined in soundSettingsChanged() and recomputeHarmonics()
        self.audio = None  # defined in soundSettingsChanged() and recomputeHarmonics()
        self.stream = None  # defined in soundSettingsChanged()
        self.pyaudio_instance = None
        self.order_widgets = []
        self.amplitude_widgets = []
        self.phase_widgets = []
        self.term_plots = []
        
        # Create diagrams
        for i in range(self.count):
            self.order_widgets.append(getattr(self, f'_order{i}'))
            self.amplitude_widgets.append(getattr(self, f'_amplitude{i}'))
            self.phase_widgets.append(getattr(self, f'_phase{i}'))
            self.term_plots.append(self._termsPlot.plot([0.0, 1.0], [1 + 2 * i, 1 + 2 * i]))
        self.sum_plot = self._sumPlot.plot([0.0, 1.0], [0.0, 0.0])
        # self.envelope_plot = self._sumPlot.plot([0.0, 1.0], [0.0, 0.0], pen=pg.mkPen('b', style=QtCore.Qt.DashLine))
        self._sumPlot.setLabel('bottom', 'Time', 's')
        self._sumPlot.setLabel('left', 'Field strength (arb. u.)')
        self._termsPlot.setLabel('bottom', 'Time', 's')
        self._termsPlot.setLabel('left', 'Field strength (arb. u.) offset by order')
        self.intensity_plot = self._intensityPlot.plot([0.0, 1.0], [0.0, 0.0], pen='b')
        self._intensityPlot.setLabel('bottom', 'Time', 's')
        self._intensityPlot.setLabel('left', 'Intensity (arb. u.)')
        self._termsPlot.setXLink(self._sumPlot)
        self._intensityPlot.setXLink(self._sumPlot)
        
        self.fft_intensity = self._fftPlot.plot([0.0, 1.0], [0.0, 0.0], pen='y')
        self._fftPlot.setLabel('bottom', 'Frequency', 'Hz')
        self._fftPlot.setLabel('left', 'Spectral intensity (arb. u.)')
        self.fft_phase = self._phasePlot.plot([0.0, 1.0], [0.0, 0.0], pen=None, symbol='o', symbolBrush='g', symbolPen=None, symbolSize=2.5)
        self.fft_phase_strong = self._phasePlot.plot([0.0, 1.0], [0.0, 0.0], pen='r', symbol='o', symbolBrush='g', symbolPen=None, symbolSize=6)
        self._phasePlot.setLabel('bottom', 'Frequency', 'Hz')
        self._phasePlot.setLabel('left', 'Spectral phase', 'rad')
        self._phasePlot.setXLink(self._fftPlot)
        
        # Connect widget signals to slots (methods) that handle them
        for widget in [*self.order_widgets, *self.amplitude_widgets, *self.phase_widgets]:
            widget.valueChanged.connect(self.recomputeHarmonics)
        self._ordersAll.clicked.connect(lambda: self.setOrders(0))
        self._ordersOdd.clicked.connect(lambda: self.setOrders(1))
        self._ordersEven.clicked.connect(lambda: self.setOrders(2))
        self._shiftDown.clicked.connect(lambda: self.shiftOrders(-1))
        self._shiftUp.clicked.connect(lambda: self.shiftOrders(1))
        self._amplitudeOnes.clicked.connect(lambda: self.setAmplitudes(1.0))
        self._amplitudeZeros.clicked.connect(lambda: self.setAmplitudes(0.0))
        self._phaseSame.clicked.connect(lambda: self.setPhases(0.0))
        self._phaseAlternating.clicked.connect(lambda: self.setPhases(np.pi))
        self._fundamental.valueChanged.connect(self.soundSettingsChanged)
        self._volume.valueChanged.connect(self.soundVolumeChanged)
        self._options.currentIndexChanged.connect(self.recomputeHarmonics)
        self._amplitudeGaussian.clicked.connect(self.gaussianAmplitude)
        self._amplitudeRandom.clicked.connect(self.randomizeAmplitude)
        self._phaseRandom.clicked.connect(self.randomizePhase)
        self._phaseReduce.clicked.connect(lambda: self.phaseShift(-2 * np.pi / self.phase_widgets[i].maximum()))
        self._phaseIncrease.clicked.connect(lambda: self.phaseShift(2 * np.pi / self.phase_widgets[i].maximum()))
        
        self.setAmplitudes(0.5)  # also calls recomputeHarmonics()

    def updateArrays(self):
        for i in range(self.count):
            self.orders[i] = self.order_widgets[i].value()
            self.amplitudes[i] = self.amplitude_widgets[i].value() / self.amplitude_widget_factor
            self.phases[i] = 2 * np.pi * np.mod(self.phase_widgets[i].value() / self.phase_widgets[i].maximum() - 0.25, 1.0)

    @pyqtSlot()
    def recomputeHarmonics(self):
        self.updateArrays()
        self.single_period = self._options.currentText() == 'Gated, isolated'
        if self.result is None or (len(self.audio) == len(self.result)) == self.single_period:
            self.soundSettingsChanged()
        
        result = 0.0
        fundamental = self._fundamental.value()  # [Hz]
        coefficient = 2 * np.pi * fundamental
        for i in range(self.count):
            term = self.amplitudes[i] * np.cos(coefficient * self.orders[i] * self.time_axis + self.phases[i])
            self.term_plots[i].setData(self.time_axis, self.orders[i] + 0.5 * term)
            result += term
        
        if self.single_period:
            centre = self.time_axis.mean() - 0.5 / fundamental
            halfwidth = 0.18 / fundamental
            result *= 2**(-((self.time_axis - centre) / halfwidth / 2)**6)
            # result[np.abs(self.time_axis - centre) > 0.5 / fundamental] = 0.0
        elif self._options.currentText() == 'Slight envelope':  # Apply slight envelope on the result, which makes the FFT smoother
            centre = self.time_axis.mean()
            halfwidth = 1.0 * self.time_axis.std()
            result *= 2**(-((self.time_axis - centre) / halfwidth / 2)**2)
        elif self._options.currentText() == 'Short envelope':  # Apply envelope on the result, which makes the FFT smoother
            centre = self.time_axis.mean()
            halfwidth = 0.5 * self.time_axis.std()
            result *= 2**(-((self.time_axis - centre) / halfwidth / 2)**2)
        # else: The default option is 'Continuous waves'
        
        self.result = result
        self.recomputeAudio()  # recompute the self.audio from self.result
        self.sum_plot.setData(self.time_axis, self.result)
        
        # if envelope or single_period:
        #     self.envelope_plot.setData(self.time_axis, np.abs(result))
        # else:
        #     self.envelope_plot.setData(self.time_axis, 0.0 * result)
        # self.envelope_plot.setVisible(envelope)
        
        self.intensity_plot.setData(self.time_axis, np.abs(result) ** 2)
        
        # if mismatched_FFT_period:
        #     result = np.concatenate((np.zeros(2), result.copy(), np.zeros(2)))  # create blurring/leakage in the FFT by not using a length that is a perfect multiple of periods
        transformed = np.fft.rfft(result)
        frequencies = np.fft.rfftfreq(len(result)) / np.diff(self.time_axis[0:2])
        shown_length = min(len(transformed), self.periods_shown * (1 + self.order_widgets[0].maximum()))
        transformed = transformed[:shown_length]
        frequencies = frequencies[:shown_length]
        intensity = np.abs(transformed) ** 2
        phase = np.angle(transformed)
        self.fft_intensity.setData(frequencies, intensity)
        self.fft_phase.setData(frequencies, phase.copy())
        phase[intensity < 1E-2 * intensity.max()] = np.NaN
        phase[:-1][intensity[:-1] < intensity[1:]] = np.NaN  # not a peak, since below the following
        phase[1:][intensity[1:] < intensity[:-1]] = np.NaN  # not a peak, since below the preceeding
        self.fft_phase_strong.setData(frequencies, phase)

    def setOrders(self, which):
        """Set the harmonic order in each GUI widget.
        
        The order of the first harmonic is unchanged, or adjusted by +-1 to
        comply with the odd/even mode.
        
        Parameters
        ----------
        which : int
            Determines the offset and spacing.
            0: Consequtive orders (even and odd).
            1: Odd orders.
            2: Even orders
        """
        first_order = max(1, int(self.order_widgets[0].value()))
        if which != 0:
            first_order = ((first_order - 1) // 2) * 2 + which
        for i, widget in enumerate(self.order_widgets):
            widget.blockSignals(True)  # don't trigger an update per iteration, call recomputeHarmonics() at end
            if which == 0:  # all
                widget.setValue(first_order + i)
            else:  # only odd or even
                widget.setValue(first_order + 2 * i)
            widget.blockSignals(False)
        self.recomputeHarmonics()

    def shiftOrders(self, offset):
        """Shift the harmonic order in each GUI widget.
        
        Parameters
        ----------
        offset: int
            Each order is shifted by this amount.
        """
        first_order = max(1, int(self.order_widgets[0].value()))
        if first_order + offset < 1:
            return  # don't shift first order below 1
        for i, widget in enumerate(self.order_widgets):
            widget.blockSignals(True)  # don't trigger an update per iteration, call recomputeHarmonics() at end
            widget.setValue(max(1, widget.value() + offset))
            widget.blockSignals(False)
        self.recomputeHarmonics()

    def setAmplitudes(self, value=None, values=None):
        """Set all amplitude widgets to either the same value or an array of values."""
        for i, widget in enumerate(self.amplitude_widgets):
            widget.blockSignals(True)  # don't trigger an update per iteration, call recomputeHarmonics() at end
            widget.setValue((value if value is not None else values[i]) * self.amplitude_widget_factor)
            widget.blockSignals(False)
        self.recomputeHarmonics()

    def gaussianAmplitude(self):
        centre = self.orders.mean()
        halfwidth = 0.6 * self.orders.std()
        self.setAmplitudes(values=2**(-((self.orders - centre) / halfwidth / 2.0)**2))
        
    def randomizeAmplitude(self):
        self.setAmplitudes(values=np.random.uniform(0.0, 1.0, self.count))

    def setPhases(self, step=0.0, first=0.0):
        """Set the phases of all harmonics.
        
        Parameters
        ----------
        step: float, optional
            The phase difference between adjacent orders. The default value is 0.
        first: float, optional
            The phase difference between adjacent orders. The default value is 0.
        """
        for i, widget in enumerate(self.phase_widgets):
            widget.blockSignals(True)  # don't trigger an update per iteration, call recomputeHarmonics() at end
            phase = first + i * step  # [rad]
            widget.setValue(round(np.mod(0.25 + phase / 2 / np.pi, 1.0) * widget.maximum()))
            widget.blockSignals(False)
        self.recomputeHarmonics()

    def randomizePhase(self):
        for widget in self.phase_widgets:
            widget.blockSignals(True)  # don't trigger an update per iteration, call recomputeHarmonics() at end
            widget.setValue(np.random.randint(0, widget.maximum() + 1))
            widget.blockSignals(False)
        self.recomputeHarmonics()

    def phaseShift(self, offset, scale_by_order=None):
        if scale_by_order is None:  # auto-choice by 
            scale_by_order = (QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier)  # if shift-key was held down
        for order, widget in zip(self.orders, self.phase_widgets):
            widget.blockSignals(True)  # don't trigger an update per iteration, call recomputeHarmonics() at end
            phase = 2 * np.pi * np.mod(widget.value() / widget.maximum() - 0.25, 1.0)
            if scale_by_order:
                phase += order * offset  # [rad]
            else:
                phase += offset  # [rad]
            widget.setValue(round(np.mod(0.25 + phase / 2 / np.pi, 1.0) * widget.maximum()))
            widget.blockSignals(False)
        self.recomputeHarmonics()


    # ---- Sound


    def recomputeAudio(self):
        """Recompute the self.audio data from self.result."""
        max_result = (self.amplitudes.sum() + 2 * self.count) / 3  # weighted average between the "in-phase peak with the chosen amplitudes" and the "max possible if all amplitudes set to 1"
        volume_coefficient = self._volume.value() / self._volume.maximum() / max_result
        self.audio[:self.result.shape[0]] = (self.result * volume_coefficient).astype(np.float32)  # pyaudio doesn't support float64

    @pyqtSlot()
    def soundSettingsChanged(self):
        number_of_samples = round(self.periods_shown * self.sound_sampling_frequency / self._fundamental.value())
        self.time_axis = np.arange(number_of_samples, dtype=np.float64) / self.sound_sampling_frequency  # [s]
        self.result = np.zeros(number_of_samples, dtype=np.float64)
        if self.single_period:
            # Make longer gaps between audio pulses than shown in diagram
            self.audio = np.zeros(15 * number_of_samples, dtype=np.float32)  # pyaudio doesn't support float64
        else:
            self.audio = np.zeros(number_of_samples, dtype=np.float32)  # pyaudio doesn't support float64

        self.recomputeHarmonics()  # also calls recomputeAudio()
        
        self.stopAudio()
        if self._volume.value() > 0:
            if self.pyaudio_instance is None:
                self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32, rate=int(self.sound_sampling_frequency), channels=1,
                input=False, output=True, frames_per_buffer=self.audio.shape[0],
                stream_callback=self.pyaudio_callback, start=True)
    @pyqtSlot()
    def soundVolumeChanged(self):
        if self._volume.value() == 0:
            self.recomputeAudio()
            self.stopAudio()
        elif self.stream is None:
            self.soundSettingsChanged()
        else:  # keep playing, updated volume for next iteration of pyaudio_callback()
            self.recomputeAudio()

    @pyqtSlot()
    def stopAudio(self):
        if self.stream is not None:
            self.stream.stop_stream()
            time.sleep(self.time_axis[-1])
            self.stream.close()
            time.sleep(0.1)
            self.stream = None

    def pyaudio_callback(self, input_data, frame_count, time_info, status):
        """Produce audio data for PyAudio playback.
        
        In callback mode, PyAudio will call a specified callback function whenever 
        it needs new audio data (to play) and/or when there is new (recorded) 
        audio data available. Note that PyAudio calls the callback function in 
        a separate thread. 
        
        Parameters
        ----------
        input_data : bytes or None
            Containing frame_count frames of recorded audio data.
            The format, packing and number of channels used by the buffers are 
            determined by parameters to PyAudio.open().
        frame_count : int
            The number of frames of data in input_data and to be provided in output_data.
            Usually matches the frames_per_buffer argument to stream = pyaudio.open(...),
            which defaults to 1024.
        time_info : dict
            with the keys 'input_buffer_adc_time', 'current_time', and 'output_buffer_dac_time'
            See the PortAudio documentation for their meanings.
            Seems possible that they are all zero, e.g. for some drivers (pulseaudio).
        status : int
            Bitmask indicating whether input and/or output buffers have been inserted 
            or will be dropped to overcome underflow or overflow conditions.
            Usually 0 when everything is OK. Problems are indicated by the flags
            pyaudio.paInputUnderflow, pyaudio.paInputOverflow, pyaudio.paOutputUnderflow,
            pyaudio.paOutputOverflow and pyaudio.paPrimingOutput.
        
        Returns
        -------
        output_data : bytes or numpy.ndarray with suitable dtype
            Containing frame_count frames of returned audio data.
            The format, packing and number of channels used by the buffers are 
            determined by parameters to PyAudio.open().
        flag : pyaudio.paContinue, pyaudio.paComplete or pyaudio.paAbort
            signifying whether there are more frames to play/record
        """
        return (self.audio[:frame_count], pyaudio.paContinue)
    
    def closeEvent(self, event):
        self.stopAudio()
        super().closeEvent(event)


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
        window = HarmonicsGUI()
        window.show()
        window.raise_()  # bring window to front, useful when starting in Spyder
        sys.exit(app.exec_())
    
    start_app()
