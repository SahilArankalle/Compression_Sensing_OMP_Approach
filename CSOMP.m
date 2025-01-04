% Signal Setup
n = 2000; % Length of the signal
x_true = zeros(n, 1); % Initialize sparse signal as zeros
num_non_zero = 123; % Number of non-zero entries
non_zero_indices = randperm(n, num_non_zero);
x_true(non_zero_indices) = randn(num_non_zero, 1); % Assign random non-zero values to selected indices

% Nyquist-Shannon Sampling (Dense Signal)
% Dynamically estimate the signal bandwidth
signal_spectrum = abs(fft(x_true)); % FFT of the sparse signal
significant_indices = find(signal_spectrum > 0.01 * max(signal_spectrum)); % Threshold for significant components
max_freq_component = max(significant_indices) / n; % Normalized frequency
nyquist_rate_dynamic = 2 * max_freq_component * n; % Nyquist rate in Hz
disp(['Dynamic Nyquist Rate: ', num2str(nyquist_rate_dynamic), ' Hz']);

% Create Random Sampling Matrix for Compressed Sensing
num_samples = 400; % Number of measurements (samples) for compressed sensing
A = randn(num_samples, n); % Random measurement matrix
y = A * x_true; % Compressed measurements

% Apply Orthogonal Matching Pursuit (OMP) for Sparse Recovery
sparsity_level = num_non_zero; % Level of sparsity (number of non-zero elements)
hat_x = omp(y, A, sparsity_level);

% Reconstruction Error
error = norm(x_true - hat_x) / norm(x_true);
disp(['Reconstruction Error: ', num2str(error)]);

% Plot Original, Compressed, and Reconstructed Signals
figure;
subplot(3, 1, 1);
stem(x_true, 'b');
title('Original Sparse Signal');
xlabel('Index'); ylabel('Amplitude');
subplot(3, 1, 2);
stem(y, 'k');
title('Compressed Signal (Measurements)');
xlabel('Index'); ylabel('Amplitude');
subplot(3, 1, 3);
stem(hat_x, 'r');
title('Reconstructed Signal using OMP');
xlabel('Index'); ylabel('Amplitude');

% Sampling Rate Comparison
compressed_sampling_rate = num_samples / n * nyquist_rate_dynamic; % Reduced sampling rate after compressed sensing

% Plot Sampling Rate Comparison Graph
figure;
bar([nyquist_rate_dynamic, compressed_sampling_rate], 'FaceColor', 'b');
set(gca, 'XTickLabel', {'Original Signal (Dynamic Nyquist)', 'Compressed Signal'});
ylabel('Sampling Rate (Hz)');
title('Dynamic Sampling Rate Comparison: Original vs Compressed Signal');
grid on;

% Explain Efficiency Improvement
disp('Compressed sensing significantly reduces the sampling rate while maintaining signal reconstruction accuracy.');

% OMP Function Definition (Move this to the end of the script)
function hat_x = omp(y, A, sparsity_level)
    % Initialize variables
    residual = y;
    indices = [];
    hat_x = zeros(size(A, 2), 1);
    
    for i = 1:sparsity_level
        % Compute correlations between residual and columns of A
        correlations = abs(A' * residual);
        [~, index] = max(correlations); % Find index of maximum correlation
        
        % Augment the set of selected indices
        indices = [indices, index];
        
        % Solve least squares problem with selected columns
        A_selected = A(:, indices);
        x_selected = A_selected \ y;
        
        % Update residual
        residual = y - A_selected * x_selected;
    end
    
    % Final estimate of sparse signal
    hat_x(indices) = x_selected;
end
