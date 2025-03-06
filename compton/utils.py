import numpy as np

# DATA READER

class MCAData:
    time_idx = 9
    data_idx = 12

    def _read_data(self,data_file):
        bins = np.array([])
        counts = np.array([])

        with open(data_file,'r') as f:
            for i,line in enumerate(f.readlines()):
                if i == self.time_idx:
                    count_time, total_time = np.array(line.split(),dtype=float)
                elif i >= self.data_idx:
                    if line.startswith('$'):
                        break
                    bins = np.append(bins,i-self.data_idx)
                    counts = np.append(counts,float(line))
        
        if counts.size != 2048:
            raise ValueError('Incorrect number of bins')

        return bins, counts, count_time, total_time
    
    def _kde_smooth_data(self,bins,counts,bw=1,remove_zeros=False):
        if remove_zeros:
            bins = bins[counts > 0]
            counts = counts[counts > 0]

        bins_matrix = np.vstack([bins] * bins.size)

        density_matrix = 1/(np.sqrt(2*np.pi)*bw) * np.exp(-1/2 * ((bins_matrix - bins_matrix.T) / bw)**2)
        
        kde = np.dot(density_matrix.T,counts) / counts.sum()
        kde_err = np.dot(density_matrix.T,np.sqrt(counts)) / counts.sum()

        return bins, kde, kde_err

    def _derivative_peaks_find(self,bins,counts,dx=1):
        pass