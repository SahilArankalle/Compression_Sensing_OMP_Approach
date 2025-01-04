original_image = imread('1.jpg');
original_image = rgb2gray(original_image);
original_image = imresize(original_image, [64, 64]);
original_image = double(original_image) / 255;
[nx, ny] = size(original_image);

dct_image = dct2(original_image);
x_true = dct_image(:);
n = length(x_true);

num_samples = round(0.6 * n);
A = randn(num_samples, n);
y = A * x_true;

sparsity_level = round(0.1 * n);
hat_x = omp(y, A, sparsity_level);
reconstructed_dct = reshape(hat_x, nx, ny);

reconstructed_image = idct2(reconstructed_dct);
reconstructed_image = (reconstructed_image - min(reconstructed_image(:))) / ...
                      (max(reconstructed_image(:)) - min(reconstructed_image(:)));

error = norm(x_true - hat_x) / norm(x_true);

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

disp(['Reconstruction Error: ', num2str(error)]);

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
