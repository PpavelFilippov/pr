import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate, peak_widths
import os

os.chdir(os.getcwd())

class SignalProcessor:
    def __init__(self, file_path_1, file_path_2, column_index):
        self.file_path_1 = file_path_1
        self.file_path_2 = file_path_2
        self.column_index = column_index
        self.signals_4 = None
        self.signal_1_fib = None
        self.mean_signal = None
        self.scaled_reference = None
        self.combined_signal = None
        self.filtered_signal = None
        self.aligned_signals = None
        self.reference_signal = None
        self.drop_point = None
        self.load_data() 

    def load_data(self):
        data_1 = pd.read_csv(self.file_path_1)
        data_2 = pd.read_csv(self.file_path_2)
        signals_4_fib = data_2.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        signals_1_fib = data_1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.signal_1_fib = signals_1_fib[:, self.column_index] 
        self.reference_signal = signals_4_fib[:, 0]
        self.aligned_signals = self.align_signals(signals_4_fib)

    def align_signals(self, signals):
        aligned_signals = []
        for signal in signals.T:
            correlation = correlate(signal, self.reference_signal)
            shift = np.argmax(correlation) - (len(signal) - 1)
            aligned_signal = np.roll(signal, -shift)
            aligned_signals.append(aligned_signal)
        return np.array(aligned_signals).T

    def calculate_mean_signal(self):
        self.mean_signal = np.mean(self.aligned_signals, axis=1)
        
    def find_main_peak(self, signal):
        peaks, _ = find_peaks(signal, height=np.max(signal) * 0.5)
        if peaks.size > 0:
            return peaks[np.argmax(signal[peaks])]
        else:
            raise ValueError("No")

    def find_peak_by_correlation(self, signal, reference):
        correlation = correlate(signal, reference)
        shift = np.argmax(correlation) - len(reference) 
        return shift, np.max(correlation)

    def scale_reference_signal(self):
        shift, _ = self.find_peak_by_correlation(self.signal_1_fib, self.mean_signal)
        mean_signal_shifted = np.roll(self.mean_signal, shift)
        main_peak_mean = self.find_main_peak(mean_signal_shifted)
        
     # Найти пик signal_1_fib в окрестности main_peak_mean
        search_range = 10  # Можно настроить этот диапазон по необходимости
        start = max(0, main_peak_mean - search_range)
        end = min(len(self.signal_1_fib), main_peak_mean + search_range)
        local_peak_1_fib = np.argmax(self.signal_1_fib[start:end]) + start

        max_amplitude_1_fib = self.signal_1_fib[local_peak_1_fib]
        scale_factor = max_amplitude_1_fib / mean_signal_shifted[main_peak_mean] 
        self.scaled_reference = mean_signal_shifted * scale_factor


    def find_correlation_drop_point(self, signal, reference):
        difference = np.abs(signal - reference)

        main_peak_reference = self.find_main_peak(reference)

        for i in range(main_peak_reference + 1, len(difference)):
            if difference[i] > 10: 
                return i 
        return len(difference) - 1
    
    # def find_correlation_drop_point(self, signal, reference):
    #     # Нормализуем значения сигналов для лучшего сравнения
    #     normalized_signal = signal / np.max(signal)
    #     normalized_reference = reference / np.max(reference)

    #     main_peak_reference = self.find_main_peak(normalized_reference)

    #     for i in range(main_peak_reference + 10, len(normalized_signal)):
    #         # Ищем точку, где значения normalized_reference начинают падать, а normalized_signal расти
    #         if normalized_reference[i] < normalized_reference[i - 1] and normalized_signal[i] > normalized_signal[i - 1]:
    #             return i

    #     return len(normalized_signal) - 1  # Возвращаем последний индекс, если не нашли точку падения
    
    def combine_drop(self):
        self.drop_point = self.find_correlation_drop_point(self.signal_1_fib, self.scaled_reference)
        
        weights_signal= np.ones_like(self.signal_1_fib) * 1
        weights_reference = np.ones_like(self.signal_1_fib) * 0
        if self.drop_point > 0:
            weights_signal[self.drop_point:] = 0 
            weights_reference[self.drop_point:] = 1
            

        self.combined_signal = (self.signal_1_fib * weights_signal + self.scaled_reference * weights_reference) 

    def combine_signals(self):
        self.combined_signal = self.signal_1_fib + self.scaled_reference

    def analyze_peaks(self, signal, signal_name):
        peak_positions, _ = find_peaks(signal, height=np.max(signal) * 0.5)
        peak_heights = signal[peak_positions]
        peak_widths_result = peak_widths(signal, peak_positions, rel_height=0.5)
        peak_widths_values = peak_widths_result[0]
        
        return pd.DataFrame({
            'Signal': signal_name,
            'Peak Position': peak_positions,
            'Peak Height': peak_heights,
            'Peak Width': peak_widths_values
        })


    def run_analysis(self):
        self.calculate_mean_signal()
        self.scale_reference_signal()
        self.combine_drop()

        mean_signal_peaks = self.analyze_peaks(self.mean_signal, 'Mean Signal')
        original_peaks = self.analyze_peaks(self.signal_1_fib, 'Original Signal')
        scaled_reference_peaks = self.analyze_peaks(self.scaled_reference, 'Scaled Reference Signal')
        combined_signal_peaks = self.analyze_peaks(self.combined_signal, 'Combined Signal')

        peak_df = pd.concat([mean_signal_peaks, original_peaks, scaled_reference_peaks, combined_signal_peaks])

        print(peak_df)

    def plot_signals(self):
        
        #Mean Signal
        plt.figure()
        plt.plot(self.reference_signal, label='Mean Signal (Reference)')
        plt.title('Mean Signal from 1ch_4_fib.csv')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(460, 560)
        plt.show()

        #Original Signal
        plt.figure()
        plt.plot(self.signal_1_fib, label='Original Signal from 1ch_1_fib.csv')
        plt.title('Original Signal from 1ch_1_fib.csv')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(507, 590)
        plt.show()

        #Scaled Reference Signal
        plt.figure()
        plt.plot(self.scaled_reference, label='Scaled Reference Signal')
        plt.title('Scaled Reference Signal')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(507, 590)
        plt.show()

        #Combined Signal with Original and Scaled Reference
        plt.figure()
        plt.plot(self.signal_1_fib, label='Original Signal')
        plt.plot(self.scaled_reference, label='Scaled Reference Signal')
        plt.plot(self.combined_signal, label='Combined Signal')
        plt.title('Combined Signal with Original and Scaled Reference')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(507, 590)
        plt.show()

        #Combined Signal (Full View)
        plt.figure()
        plt.plot(self.combined_signal, label='Combined Signal (Full View)')
        plt.title('Combined Signal (Full View)')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(507, 590)
        plt.show()

if __name__ == "__main__":
    file_path_1 = '1ch_1_fib.csv'
    file_path_2 = '1ch_4_fib.csv'
    column_index = 146
    
    processor = SignalProcessor(file_path_1, file_path_2, column_index)
    processor.run_analysis()
    processor.plot_signals()
