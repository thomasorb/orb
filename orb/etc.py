from orb.core import Tools
import orb.utils.spectrum

class ETC(Tools):

    def __init__(self, spectrum_phys, step, order, filter_name,
                 add_sky=True, airmass=1, **kwargs):
        """Initialize class
        
        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)

        self.step_nb = 500 # any value is ok
        # correction value at the center of the frame
        self.axis_corr = 1. / np.cos(float(
            self._get_config_parameter('OFF_AXIS_ANGLE_CENTER'))) 
        self.airmass = airmass
        self.step = float(step)
        self.order = int(order)
        self.filter_name = str(filter_name)
        self.filter_file = FilterFile(self.filter_name)

        self.cm1_axis = orb.utils.spectrum.create_cm1_axis(
            self.step_nb, self.step, self.order, corr=self.axis_corr)

        self._nm_axis = orb.utils.spectrum.create_nm_axis(
            self.step_nb, self.step, self.order, corr=self.axis_corr)
        
        # convert spectrum units from erg/cm2/s/A to counts/A
        self.spectrum_phys = spectrum_phys # erg/cm2/s/A
        
        if add_sky:
            self.sky_spectrum = self.get_sky_spectrum()
        else:
            self.sky_spectrum = 0.
            
        self.spectrum_phys += self.sky_spectrum # erg/cm2/s/A

        self.spectrum_counts = self.convert(self.spectrum_phys) # counts/A/s
        self.interferogram = FFT(self.spectrum_counts) # counts/s
        # also apply modulation efficiency's effect to interferogram

    def simulate_spectrum(self, exposition_time, remove_sky=True):

        # convert counts/s > counts
        interferogram *= exposition_time
        
        # add noise
        interferogram_noise = interferogram + noise
        
        # FFT
        spectrum_sim_noise = FFT(interferogram_noise)
        spectrum_sim = FFT(interferogram)

        if remove_sky:
            # !!! sky spectrum used for subtraction should be obtained through simulation
            # to obtain correct unit scale and line fwhm
            spectrum_sim -= self.sky_spectrum
            spectrum_sim_noise -= self.sky_spectrum
            
        return spectrum_sim, spectrum_sim_noise # counts


    def get_sky_spectrum(self):
        # mail @ lison pour avoir un spectre du ciel de Espadon
        return 0

    def convert(self):

        # convert from erg/cm2/s/A to counts/A/s

        # wavenumber (or wavelength) dependant curves.
        # given on the same wavenumber axis as the spectrum :
        # quantum efficiency, filter transmission (or absorption),
        # atmospheric extinction given in function of airmass
        # and additionnal optics transmission

        spectrum_counts = np.copy(self.spectrum_phys)
        spectrum_counts /= orb.utils.photometry.compute_photon_energy(
            self._nm_axis)
        spectrum_counts *= orb.utils.photometry.get_atmospheric_transmission(
            self._get_atmospheric_extinction_file_path(),
            self.step, self.order, self.step_nb,
            airmass=self.airmass, corr=self.axis_corr)
        spectrum_counts *= get_config_parameter('MIR_SURFACE') # photons/s/A
        spectrum_counts *= orb.utils.photometry.get_mirror_transmission(
            self._get_mirror_transmission_file_path(),
            self.step, self.order, self.step_nb, corr=self.axis_corr)**2 # **2 because we have two mirrors 
        spectrum_counts *= orb.utils.photometry.get_optics_transmission(
            self._get_optics_file_path(self.filter_name),
            self.step, self.order, self.step_nb, corr=self.axis_corr)
        spectrum_counts *= self.filterfile.get_filter_function(
            self.step, self.order, self.step_nb,
            wavenumber=True, corr=self.axis_corr)[0]
        # must be modified for SpIOMM to use camera 1 or camera 2
        spectrum_counts *= orb.utils.photometry.get_quantum_efficiency(
            _get_quantum_efficiency_file_path(1),
            self.step, self.order, self.step_nb, corr=self.axis_corr) # electrons/s/A
        spectrum_counts *= self.get_config_parameter("CAM1_GAIN") # counts/s/A
    

        # eliminate cm2, split light between two ccds
        mirror_area = float(self._get_config_parameter('MIR_SURFACE'))
        spectrum *= mirror_area * 0.5

        return spectrum

    def get_snr(self, exposition_time):

        # simulate 10000 times
        for i in range(10000):
            ispectrum_sim, ispectrum_sim_noise = simulate_spectrum(
                exposition_time)
            inoise = ispectrum_sim_noise - ispectrum_sim
            
        # get mean RMS 
        mean_rms = np.std(noise_list)
    
        return ispectrum_sim / mean_rms


    def get_exposition_time(self, target, wavenumber):

        # wavenumber can be optimized by a fit on the emission line
        
        while snr != target:
            snr = get_snr(exposition_time)[wavenumber]
            # correct exposition time in function of obtained snr before looping again
            exposition_time *= (target / snr)**2.

    def effective_exposition_time(self, exposition_time, percent, reverse=False):

        # simulated interferogram is always symmetrical
        # give the real exposure time for a non-symmetrical observation (reduced number of steps
        # by a certain amount) to obtain the effective exposure time used in the simulation
        # percent is the percentage of steps in the real interferogram compared to a symmetrical one

        if reverse:
            return exposition_time/percent # convert time obtained in simulation to real time
        else:
            return exposition_time*percent # convert real time to simulated time
