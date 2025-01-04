% ======================================================================= %
% Sparse Image Reconstruction Using Compressed Sensing in DCT Domain
% ======================================================================= %

% Load and Prepare Image
original_image = imread('1.jpg'); % Replace with your .jpg image
original_image = rgb2gray(original_image); % Convert to grayscale if needed
original_image = imresize(original_image, [64, 64]); % Resize for simplicity
original_image = double(original_image) / 255; % Normalize pixel values
[nx, ny] = size(original_image); % Image dimensions
disp('Original image loaded, converted to grayscale, and resized.');

% Transform Image to DCT Domain
dct_image = dct2(original_image); % Apply 2D Discrete Cosine Transform (DCT)
x_true = dct_image(:); % Vectorize DCT coefficients
n = length(x_true);

% ----------------------------------------------------------------------- %
% Random Sampling Matrix
num_samples = round(0.6 * n); % Use 25% of measurements for compressed sensing
A = randn(num_samples, n); % Random measurement matrix
y = A * x_true; % Compressed measurements
disp(['Compressed measurements collected: ', num2str(num_samples)]);

% ----------------------------------------------------------------------- %
% Sparse Recovery Using Orthogonal Matching Pursuit (OMP)
sparsity_level = round(0.1 * n); % Assume 10% sparsity level
hat_x = omp(y, A, sparsity_level); % Reconstruct sparse DCT coefficients
reconstructed_dct = reshape(hat_x, nx, ny); % Reshape to 2D DCT coefficients

% Inverse DCT to Recover the Image
reconstructed_image = idct2(reconstructed_dct); % Reconstruct image from DCT coefficients

% Normalize the Reconstructed Image for Display
reconstructed_image = (reconstructed_image - min(reconstructed_image(:))) / ...
                      (max(reconstructed_image(:)) - min(reconstructed_image(:)));

% ----------------------------------------------------------------------- %
% Reconstruction Error
error = norm(x_true - hat_x) / norm(x_true); % Normalized error
disp(['Reconstruction Error: ', num2str(error)]);

% ----------------------------------------------------------------------- %
% Visualization
figure;
subplot(1, 3, 1);
imshow(original_image, []);
title('Original Image', 'FontSize', 12);

subplot(1, 3, 2);
imshow(reshape(A' * y, nx, ny), []);
title('Compressed Representation', 'FontSize', 12);

subplot(1, 3, 3);
imshow(reconstructed_image, []);
title('Reconstructed Image (DCT Domain)', 'FontSize', 12);

% ----------------------------------------------------------------------- %
% Efficiency Statement
disp('Compressed sensing reduces image sampling while preserving reconstruction accuracy.');

% ----------------------------------------------------------------------- %
% OMP Function Definition
function hat_x = omp(y, A, sparsity_level)
    % Orthogonal Matching Pursuit (OMP) Algorithm
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
