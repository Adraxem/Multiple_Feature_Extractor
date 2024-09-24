# Multiple_Feature_Extractor
Road Surface Data Analysis Using Acceleration and Pitch
This project is focused on analyzing road surface conditions through Acc_x, Acc_y, Acc_z datasets using Python. The data is collected from various road conditions and categorized into cavity and bump types, based on acceleration measurements.

Key Features:
Data Extraction: Extracts specific columns (Acc_x, Acc_y, Acc_z) from raw CSV files.
Signal Analysis: Performs a variety of analyses, including:
Moving averages for smoothing
Gradient calculations
Crossings (number of times signals cross a threshold)
Statistical features (mean, max, min, standard deviation, skewness, kurtosis)
Frequency-domain analysis using FFT (Fast Fourier Transform)
Autocorrelation analysis for detecting periodicity and peak detection
Cavity and Bump Classification: Classifies and analyzes road segments into "cavity" or "bump" based on predefined data patterns.
CSV Result Generation: Automatically saves results to CSV files for further analysis and visualization.
Dependencies:
numpy for efficient numerical computations
scipy for FFT, skewness, and kurtosis computations
matplotlib for future visualizations
csv for data manipulation and saving results
How It Works:
The program first reads raw road data from CSV files.
Extracts specific columns (e.g., acceleration, pitch) and saves them into a new file.
Applies signal processing techniques like moving averages and gradient calculation.
Performs frequency-domain analysis using FFT.
Saves all analysis results, including statistical summaries, FFT metrics, and more, into CSV files.
Output:
The code generates a CSV file with detailed analysis, including:

Mean, max, and min values of Acc_z and Pitch signals
FFT results for frequency-domain analysis
Signal crossings and gradient extrema
Skewness, kurtosis, and other statistical features
This analysis can be used to detect and classify different road conditions based on sensor readings from road vehicles.
