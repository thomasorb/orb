from orb.core import Tools


class ETC(Tools):

    def __init__(self, spectrum_phys, step, order, filter_name,
                 add_sky=True, **kwargs):
        """Initialize class
        
        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)

        self.step = float(step)
        self.order = int(order)
        self.filter_name = str(filter_name)

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

        # eliminate cm2, split light between two ccds
        mirror_area = float(self._get_config_parameter('MIR_SURFACE'))
        spectrum *= mirror_area * 0.5

        # wavenumber (or wavelength) dependant curves.
        # given on the same wavenumber axis as the spectrum :
        # quantum efficiency, filter transmission (or absorption),
        # atmospheric extinction given in function of airmass
        # and additionnal optics transmission
        corr = float(self._get_config_parameter('OFF_AXIS_ANGLE_CENTER')) ; quit()
        (filter_trans,
         filter_min, filter_max) = FilterFile(self.filter_name).get_filter_function(
            self.step, self.order, STEP_NB, corr=corr)
        
        spectrum *= quantum_efficiency * filter_transmission * atmospheric_extinction * optics_transmission

        # convert ergs to photon count
        spectrum = erg_to_count(spectrum) #using E=h*nu

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
            exposition_time *= (target / snr) ^ 2

    def effective_exposition_time(self, exposition_time, percent, reverse=False):

        # simulated interferogram is always symmetrical
        # give the real exposure time for a non-symmetrical observation (reduced number of steps
        # by a certain amount) to obtain the effective exposure time used in the simulation
        # percent is the percentage of steps in the real interferogram compared to a symmetrical one

        if reverse:
            return exposition_time/percent # convert time obtained in simulation to real time
        else:
            return exposition_time*percent # convert real time to simulated time
