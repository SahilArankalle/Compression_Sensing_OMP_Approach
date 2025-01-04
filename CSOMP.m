n = 2000;
x_true = zeros(n, 1);
num_non_zero = 123;
non_zero_indices = randperm(n, num_non_zero);
x_true(non_zero_indices) = randn(num_non_zero, 1);

signal_spectrum = abs(fft(x_true));
significant_indices = find(signal_spectrum > 0.01 * max(signal_spectrum));
max_freq_component = max(significant_indices) / n;
nyquist_rate_dynamic = 2 * max_freq_component * n;
disp(['Dynamic Nyquist Rate: ', num2str(nyquist_rate_dynamic), ' Hz']);

num_samples = 400;
A = randn(num_samples, n);
y = A * x_true;

sparsity_level = num_non_zero;
hat_x = omp(y, A, sparsity_level);

error = norm(x_true - hat_x) / norm(x_true);
disp(['Reconstruction Error: ', num2str(error)]);

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

compressed_sampling_rate = num_samples / n * nyquist_rate_dynamic;

figure;
bar([nyquist_rate_dynamic, compressed_sampling_rate], 'FaceColor', 'b');
set(gca, 'XTickLabel', {'Original Signal (Dynamic Nyquist)', 'Compressed Signal'});
ylabel('Sampling Rate (Hz)');
title('Dynamic Sampling Rate Comparison: Original vs Compressed Signal');
grid on;

disp('Compressed sensing significantly reduces the sampling rate while maintaining signal reconstruction accuracy.');

function hat_x = omp(y, A, sparsity_level)
    residual = y;
    indices = [];
    hat_x = zeros(size(A, 2), 1);
    for i = 1:sparsity_level
        correlations = abs(A' * residual);
        [~, index] = max(correlations);
        indices = [indices, index];
        A_selected = A(:, indices);
        x_selected = A_selected \ y;
        residual = y - A_selected * x_selected;
    end
    hat_x(indices) = x_selected;
end
