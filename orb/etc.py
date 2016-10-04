class ETC:

    def __init__(self, spectrum_phys, step, order, add_sky=True):

        # convert spectrum units from erg/cm2/s/A to counts/A
        self.spectrum_phys = spectrum_phys # erg/cm2/s/A
        
        if add_sky:
            self.sky_spectrum = self.get_sky_spectrum()
            self.spectrum_phys += self.sky_spectrum # erg/cm2/s/A
        else:
            self.sky_spectrum = 0.
        
        self.spectrum_counts = self.convert(self.spectrum_phys) # counts/A/s
        self.interferogram = FFT(self.spectrum_counts) # counts/s
        
    def simulate_spectrum(self, exposition_time, remove_sky=True):

        # convert counts/s > counts
        interferogram *= exposition_time
        
        # add noise
        interferogram_noise = interferogram + noise
        
        # FFT
        spectrum_sim_noise = FFT(interferogram_noise)
        spectrum_sim = FFT(interferogram)

        if remove_sky:
            spectrum_sim -= self.sky_spectrum
            spectrum_sim_noise -= self.sky_spectrum
            
        return spectrum_sim, spectrum_sim_noise # counts
    


    def get_sky_spectrum(self):
        # mail @ lison pour avoir un spectre du ciel de Espadon
        pass

    def convert(self):
        pass

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
        
        while snr < target:
            snr = get_snr(exposition_time)[wavenumber]
