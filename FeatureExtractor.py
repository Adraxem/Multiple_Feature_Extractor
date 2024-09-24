import os
import csv
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

def extract_columns(input_file, output_file, columns):
    with open(input_file, 'r') as csv_in:
        reader = csv.reader(csv_in)
        data = list(reader)
        extracted_columns = [[data[0][col]] for col in columns]  # Extract headers first
        for row in data[1:]:
            for i, col in enumerate(columns):
                extracted_columns[i].append(row[col])  # Append each extracted column
        with open(output_file, 'a', newline='') as csv_out:
            writer = csv.writer(csv_out)
            for i, extracted_column in enumerate(extracted_columns):
                writer.writerow([f'{os.path.basename(input_file)} column{i+1}'] + extracted_column)
            writer.writerow([])  # Add an empty row as separator


def extract_rows_data(csv_file, row_indices):
    rows_data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row_index in row_indices:
            if row_index < len(rows):
                row_data = rows[row_index][2:]  # Exclude filename and column index from row data
                row_data = [float(val) for val in row_data]  # Convert data to floats
                rows_data.append(row_data)
    return rows_data


def analyze_data(data, label):

    analysis_results = []
    for row_data in range(0,len(data),2):

        # Calculate moving averages
        window_size_ma = 51
        row_data_ma_acc_z = data[row_data]
        row_data_ma_pitch = data[row_data+1]

        row_data_mean_acc_z = np.mean(data[row_data])
        row_data_max_acc_z = np.max(data[row_data])
        row_data_min_acc_z = np.min(data[row_data])

        row_data_mean_pitch = np.mean(data[row_data+1])
        row_data_max_pitch = np.max(data[row_data+1])
        row_data_min_pitch = np.min(data[row_data+1])

        grad_acc_z_max = np.max(np.absolute(np.gradient(row_data_ma_acc_z)))
        grad_acc_z_min = np.min(np.absolute(np.gradient(row_data_ma_acc_z)))

        grad_pitch_max = np.max(np.absolute(np.gradient(row_data_ma_pitch)))
        grad_pitch_min = np.min(np.absolute(np.gradient(row_data_ma_pitch)))

        crossings_acc_z = 0
        previous_value = data[row_data][0]
        for value in row_data_ma_acc_z[1:]:
            if (value >= 11 and previous_value < 11) or (value < 11 and previous_value >= 11): #15-crossings
                crossings_acc_z += 1
            previous_value = value

        crossings_pitch = 0
        previous_value = data[row_data+1][0]
        for value in row_data_ma_pitch[1:]:
            if (value >= 0 and previous_value < 0) or (value < 0 and previous_value >= 0): #15-crossings
                crossings_pitch += 1
            previous_value = value

        std_ma_acc_z = np.std(row_data_ma_acc_z)
        kurtosis_ma_acc_z = kurtosis(row_data_ma_acc_z)
        skewness_ma_acc_z = skew(row_data_ma_acc_z)

        std_ma_pitch = np.std(row_data_ma_pitch)
        kurtosis_ma_pitch = kurtosis(row_data_ma_pitch)
        skewness_ma_pitch = skew(row_data_ma_pitch)

        # Calculate the FFT of the row data
        fft_row_data_acc_z = fft(row_data_ma_acc_z)
        fft_row_data_pitch = fft(row_data_ma_pitch)

        autocorr_acc_z = np.correlate(row_data_ma_acc_z, row_data_ma_pitch, mode='full')
        autocorr_acc_z = autocorr_acc_z[len(autocorr_acc_z) // 2:]  # Take only second half of autocorrelation

        max_autocorr_idx_acc_z = np.argmin(autocorr_acc_z)

        reduced_number_acc_z = row_data_ma_acc_z[max_autocorr_idx_acc_z]

        autocorr_pitch = np.correlate(row_data_ma_pitch, row_data_ma_pitch, mode='full')
        autocorr_pitch = autocorr_pitch[len(autocorr_pitch) // 2:]  # Take only second half of autocorrelation

        max_autocorr_idx_pitch = np.argmin(autocorr_pitch)

        reduced_number_pitch = row_data_ma_pitch[max_autocorr_idx_pitch]



        max_fft_acc_z = np.max(np.abs(fft_row_data_acc_z))
        min_fft_acc_z = np.min(np.abs(fft_row_data_acc_z))
        mean_fft_acc_z = np.mean(np.abs(fft_row_data_acc_z))

        max_fft_pitch = np.max(np.abs(fft_row_data_pitch))
        min_fft_pitch = np.min(np.abs(fft_row_data_pitch))
        mean_fft_pitch = np.mean(np.abs(fft_row_data_pitch))

        analysis_results.append([row_data_mean_acc_z, row_data_mean_pitch,
                                 row_data_max_acc_z, row_data_max_pitch,
                                 row_data_min_acc_z, row_data_min_pitch,
                                 grad_acc_z_max, grad_acc_z_min, grad_pitch_max,grad_pitch_min,
                                 skewness_ma_acc_z, skewness_ma_pitch,reduced_number_acc_z,reduced_number_pitch,
                                 max_fft_acc_z, min_fft_acc_z,mean_fft_acc_z, max_fft_pitch,
                                  min_fft_pitch, mean_fft_pitch, crossings_acc_z, crossings_pitch,
                                 kurtosis_ma_acc_z, kurtosis_ma_pitch,
                                 std_ma_acc_z, std_ma_pitch, label])

    return analysis_results




def save_results_to_csv(results, output_csv):
    header = ["Raw data mean Acc_z", "Raw data Mean Pitch", "Raw data Max Acc_z", "Raw data Max Pitch", "Raw data Min Acc_z", "Raw data Min Pitch", "Gradient Max Acc_z","Gradient Min Acc_z","Gradient Max Pitch","Gradient Min Pitch", "Skewness Acc_z", "Skewness Pitch","Lag Autocorr peak Acc_z","Lag Autocorr peak Pitch", "Max FFT Acc_z","Min FFT Acc_z","Mean FFT Acc_z", "Max FFT Pitch", "Min FFT Pitch","Mean FFT Pitch","crossings acc_z","crossings pitch", "Kurtosis Acc_z","Kurtosis Pitch", "Standard Deviation Acc_z","Standard Deviation Pitch", "Label"]
    write_header = not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0
    with open(output_csv, 'a', newline='') as csv_out:
        writer = csv.writer(csv_out)
        if write_header:
            writer.writerow(header)  # Write header row
        for result in results:
            writer.writerow(result)


# Define the directories and files
data_directory = './bitir'
bump_directory = './bump'
window_directory = './generated_windows'
output_file = './output.csv'
results_csv = './results.csv'

if os.path.exists(output_file):
    os.remove(output_file)
if os.path.exists(results_csv):
    os.remove(results_csv)

# Note: Column numbering starts from 0
columns_to_extract = [3, 15]
rows_to_analyze_cavity = []
for i in range(0, 319, 3):
    rows_to_analyze_cavity.extend([i, i+1])
rows_to_analyze_bump = []
for i in range(321, 641, 3):
    rows_to_analyze_bump.extend([i, i+1])

window_labels = 0  # Label for 'window_data' directory

def analyze_window_data(input_file, label):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = np.array([list(map(float, row)) for row in reader])  # Convert all rows to floats
        analysis_results = []

        # Calculate moving averages
        acc_z_ma = data[:, 0]
        pitch_ma = data[:, 1]

        grad_acc_z_max = np.max(np.absolute(np.gradient(acc_z_ma)))
        grad_acc_z_min = np.min(np.absolute(np.gradient(acc_z_ma)))
        grad_pitch_max = np.max(np.absolute(np.gradient(pitch_ma)))
        grad_pitch_min = np.min(np.absolute(np.gradient(pitch_ma)))

        row_data_mean_acc_z = np.mean(data[:, 0])
        row_data_mean_pitch = np.mean(data[:, 1])
        row_data_max_acc_z = np.max(data[:, 0])
        row_data_max_pitch = np.max(data[:, 1])
        row_data_min_acc_z = np.min(data[:, 0])
        row_data_min_pitch = np.min(data[:, 1])


        std_acc_z = np.std(acc_z_ma)
        std_pitch = np.std(pitch_ma)
        kurtosis_acc_z = kurtosis(acc_z_ma)
        kurtosis_pitch = kurtosis(pitch_ma)
        skewness_acc_z = skew(acc_z_ma)
        skewness_pitch = skew(pitch_ma)

        # Calculate the FFT of the column data
        acc_z_fft = fft(acc_z_ma)
        pitch_fft = fft(pitch_ma)

        max_fft_acc_z = np.max(np.abs(acc_z_fft))
        max_fft_pitch = np.max(np.abs(pitch_fft))
        mean_fft_acc_z = np.mean(np.abs(acc_z_fft))
        min_fft_acc_z = np.min(np.abs(acc_z_fft))
        min_fft_pitch = np.min(np.abs(pitch_fft))
        mean_fft_pitch = np.mean(np.abs(pitch_fft))

        autocorr_acc_z = np.correlate(data[:, 0], data[:, 0], mode='full')
        autocorr_acc_z = autocorr_acc_z[len(autocorr_acc_z) // 2:]  # Take only second half of autocorrelation

        max_autocorr_idx_acc_z = np.argmin(autocorr_acc_z)

        reduced_number_acc_z = data[:, 0][max_autocorr_idx_acc_z]

        autocorr_pitch = np.correlate(data[:, 1], data[:, 1], mode='full')
        autocorr_pitch = autocorr_pitch[len(autocorr_pitch) // 2:]  # Take only second half of autocorrelation

        max_autocorr_idx_pitch = np.argmin(autocorr_pitch)

        reduced_number_pitch = data[:, 0][max_autocorr_idx_pitch]


        crossings_acc_z = 0
        previous_value = data[:, 0][0]
        for value in data[:, 0][1:]:
            if (value >= 11 and previous_value < 11) or (value < 11 and previous_value >= 11): #20-crossings
                crossings_acc_z += 1
            previous_value = value

        crossings_pitch = 0
        previous_value = data[:, 1][0]
        for value in data[:, 1][1:]:
            if (value >= 0 and previous_value < 0) or (value < 0 and previous_value >= 0): #15-crossings
                crossings_pitch += 1
            previous_value = value



        analysis_results.append([row_data_mean_acc_z, row_data_mean_pitch,
                                 row_data_max_acc_z, row_data_max_pitch,
                                 row_data_min_acc_z, row_data_min_pitch,
                                 grad_acc_z_max, grad_acc_z_min, grad_pitch_max, grad_pitch_min,
                                 skewness_acc_z, skewness_pitch, reduced_number_acc_z, reduced_number_pitch,
                                 max_fft_acc_z, min_fft_acc_z, mean_fft_acc_z, max_fft_pitch,
                                  min_fft_pitch, mean_fft_pitch, crossings_acc_z, crossings_pitch,
                                 kurtosis_acc_z, kurtosis_pitch,
                                 std_acc_z, std_pitch, label])
        """
            for row_data in range(0,len(data),2):

        # Calculate moving averages
        window_size_ma = 51
        row_data_ma_acc_z = moving_average(data[row_data], window_size_ma)
        row_data_ma_pitch = moving_average(data[row_data+1], window_size_ma)

        row_data_mean_acc_z = np.mean(data[row_data])
        row_data_max_acc_z = np.max(data[row_data])
        row_data_min_acc_z = np.min(data[row_data])

        row_data_mean_pitch = np.mean(data[row_data+1])
        row_data_max_pitch = np.max(data[row_data+1])
        row_data_min_pitch = np.min(data[row_data+1])

        grad_acc_z_max = np.max(np.absolute(np.gradient(row_data_ma_acc_z)))
        grad_acc_z_min = np.min(np.absolute(np.gradient(row_data_ma_acc_z)))

        grad_pitch_max = np.max(np.absolute(np.gradient(row_data_ma_pitch)))
        grad_pitch_min = np.min(np.absolute(np.gradient(row_data_ma_pitch)))

        crossings_acc_z = 0
        previous_value = data[row_data][0]
        for value in row_data_ma_acc_z[1:]:
            if (value >= 11 and previous_value < 11) or (value < 11 and previous_value >= 11): #15-crossings
                crossings_acc_z += 1
            previous_value = value

        crossings_pitch = 0
        previous_value = data[row_data+1][0]
        for value in row_data_ma_pitch[1:]:
            if (value >= 0 and previous_value < 0) or (value < 0 and previous_value >= 0): #15-crossings
                crossings_pitch += 1
            previous_value = value

        std_ma_acc_z = np.std(row_data_ma_acc_z)
        kurtosis_ma_acc_z = kurtosis(row_data_ma_acc_z)
        skewness_ma_acc_z = skew(row_data_ma_acc_z)

        std_ma_pitch = np.std(row_data_ma_pitch)
        kurtosis_ma_pitch = kurtosis(row_data_ma_pitch)
        skewness_ma_pitch = skew(row_data_ma_pitch)

        # Calculate the FFT of the row data
        fft_row_data_acc_z = fft(row_data_ma_acc_z)
        fft_row_data_pitch = fft(row_data_ma_pitch)

        autocorr_acc_z = np.correlate(row_data_ma_acc_z, row_data_ma_pitch, mode='full')
        autocorr_acc_z = autocorr_acc_z[len(autocorr_acc_z) // 2:]  # Take only second half of autocorrelation

        max_autocorr_idx_acc_z = np.argmin(autocorr_acc_z)

        reduced_number_acc_z = row_data_ma_acc_z[max_autocorr_idx_acc_z]

        autocorr_pitch = np.correlate(row_data_ma_pitch, row_data_ma_pitch, mode='full')
        autocorr_pitch = autocorr_pitch[len(autocorr_pitch) // 2:]  # Take only second half of autocorrelation

        max_autocorr_idx_pitch = np.argmin(autocorr_pitch)

        reduced_number_pitch = row_data_ma_pitch[max_autocorr_idx_pitch]



        max_fft_acc_z = np.max(np.abs(fft_row_data_acc_z))
        min_fft_acc_z = np.min(np.abs(fft_row_data_acc_z))
        mean_fft_acc_z = np.mean(np.abs(fft_row_data_acc_z))

        max_fft_pitch = np.max(np.abs(fft_row_data_pitch))
        min_fft_pitch = np.min(np.abs(fft_row_data_pitch))
        mean_fft_pitch = np.mean(np.abs(fft_row_data_pitch))

        analysis_results.append([row_data_mean_acc_z, row_data_mean_pitch,
                                 row_data_max_acc_z, row_data_max_pitch,
                                 row_data_min_acc_z, row_data_min_pitch,
                                 grad_acc_z_max, grad_acc_z_min, grad_pitch_max,grad_pitch_min,
                                 skewness_ma_acc_z, skewness_ma_pitch,reduced_number_acc_z,reduced_number_pitch,
                                 max_fft_acc_z, min_fft_acc_z,mean_fft_acc_z, max_fft_pitch,
                                  min_fft_pitch, mean_fft_pitch, crossings_acc_z, crossings_pitch,
                                 kurtosis_ma_acc_z, kurtosis_ma_pitch,
                                 std_ma_acc_z, std_ma_pitch, label])

    return analysis_results

        
        
        
        
        """

    return analysis_results
"""
        analysis_results.append([row_data_mean_acc_z, row_data_mean_pitch,
                                 row_data_max_acc_z, row_data_max_pitch,
                                 row_data_min_acc_z, row_data_min_pitch,
                                 grad_acc_z_max, grad_acc_z_min, grad_pitch_max,grad_pitch_min,
                                 skewness_ma_acc_z, skewness_ma_pitch,reduced_number_acc_z,reduced_number_pitch,
                                 max_fft_acc_z, min_fft_acc_z,mean_fft_acc_z, max_fft_pitch,
                                  min_fft_pitch, mean_fft_pitch, crossings_acc_z, crossings_pitch,
                                 kurtosis_ma_acc_z, kurtosis_ma_pitch,
                                 std_ma_acc_z, std_ma_pitch, label])
"""



# Define the directories and files
data_directory = './cavity'
window_directory = './generated_windows'
output_file = './output.csv'
results_csv = './results.csv'

# Note: Column numbering starts from 0
columns_to_extract = [3, 15]
window_labels = 0  # Label for 'window_data' directory


for filename in os.listdir(data_directory):
    if filename.startswith('Cavity_Data_') and filename.endswith('.csv'):
        input_file = os.path.join(data_directory, filename)
        extract_columns(input_file, output_file, columns_to_extract)
        print(f'Extracted data from {filename} and appended to {output_file}')

# Extract rows data from the output CSV file
rows_data = extract_rows_data(output_file, rows_to_analyze_cavity)


analysis_results = analyze_data(rows_data, 0)  # Label 1 for 'data' directory
save_results_to_csv(analysis_results, results_csv)
print("Analysis results from 'data' directory saved to", results_csv)


for filename in os.listdir(bump_directory):
    if filename.startswith('Cavity_Data_') and filename.endswith('.csv'):
        input_file = os.path.join(bump_directory, filename)
        extract_columns(input_file, output_file, columns_to_extract)
        print(f'Extracted data from {filename} and appended to {output_file}')

rows_data = extract_rows_data(output_file, rows_to_analyze_bump)
analysis_results = analyze_data(rows_data, 1)  # Label 2 for 'bumps' directory
save_results_to_csv(analysis_results, results_csv)
