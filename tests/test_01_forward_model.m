% test_01_forward_model.m
%
% PIECE 1: Forward Model Setup and Verification
%
% What this tests:
%   1. Phantom creation
%   2. Forward operator A construction via XTM_Tensor_XH
%   3. Sinogram generation: s = A*x + noise
%   4. Gradient check: grad_f(x) = A'*(A*x - s) at x_true
%   5. Objective value at x_true vs zeros
%
% After this test passes, we build the restriction/prolongation operators.

clear; clc; close all;

%% 0. Add paths
repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'utils'));
addpath(fullfile(repo_root, 'operators'));

fprintf('=== PIECE 1: Forward Model Test ===\n\n');

%% 1. Problem size  (keep small for fast testing)
Nx = 64; Ny = Nx;
WSz = [Ny, Nx];

NTheta = 30;                                           % number of angles
NTau   = ceil(sqrt(Nx^2 + Ny^2));                     % number of detector bins
NTau   = NTau + mod(NTau - Ny, 2);                    % match parity with Ny
SSz    = [NTheta, NTau];

fprintf('Image grid  : %d x %d  (%d unknowns)\n', Nx, Ny, Nx*Ny);
fprintf('Sinogram    : %d angles x %d bins  (%d measurements)\n', NTheta, NTau, NTheta*NTau);

%% 2. Create phantom and normalise
WGT = createObject('Phantom', WSz);        % values in [0,1]
x_true = WGT(:);                           % vectorised ground truth

%% 3. Build forward operator A
fprintf('\nBuilding forward operator A ... ');
tic;
A = XTM_Tensor_XH(WSz, NTheta, NTau, 0, WGT);
build_time = toc;
fprintf('done in %.2f s\n', build_time);
fprintf('A size: %d x %d   (sparse: %.1f%% nonzero)\n', ...
    size(A,1), size(A,2), 100*nnz(A)/numel(A));

% Normalise rows so each ray sums to ~1 (standard in tomography)
row_sums   = full(sum(A, 2));
row_sums(row_sums == 0) = 1;               % avoid divide-by-zero for empty rays
A_norm_vec = 1 ./ row_sums;
A = spdiags(A_norm_vec, 0, size(A,1), size(A,1)) * A;

%% 4. Generate clean sinogram, then add noise
s_clean = A * x_true;                     % noise-free sinogram vector

noise_std = 0.01;                          % 1% Gaussian noise
rng(42);
noise = noise_std * randn(size(s_clean));
s = s_clean + noise;                       % measured sinogram (with noise)

S_img = reshape(s, NTheta, NTau);         % 2D sinogram for display

%% 5. Verify forward model: objective and gradient

% Objective: f(x) = 0.5 * ||A*x - s||^2
f = @(x) 0.5 * norm(A*x - s)^2;

% Gradient: grad_f(x) = A' * (A*x - s)
gradf = @(x) A' * (A*x - s);

f_at_true  = f(x_true);
f_at_zeros = f(zeros(Nx*Ny, 1));

fprintf('\n--- Objective check ---\n');
fprintf('f(x_true)   = %.6e\n', f_at_true);
fprintf('f(0)        = %.6e   (should be >> f(x_true))\n', f_at_zeros);
assert(f_at_true < f_at_zeros, 'FAIL: f(x_true) should be smaller than f(0)');
fprintf('PASS: f(x_true) < f(0)\n');

% Gradient at x_true should be small (||A'*(Ax-s)|| ≈ ||A'*noise||)
grad_at_true = gradf(x_true);
grad_norm    = norm(grad_at_true);
noise_level  = norm(A' * noise);          % expected magnitude

fprintf('\n--- Gradient check at x_true ---\n');
fprintf('||grad_f(x_true)||       = %.6e\n', grad_norm);
fprintf('||A^T * noise||          = %.6e   (expected magnitude)\n', noise_level);
fprintf('Ratio (should be ~1)     = %.4f\n', grad_norm / noise_level);

rel_err = abs(grad_norm - noise_level) / noise_level;
assert(rel_err < 0.01, 'FAIL: gradient norm deviates more than 1%% from A^T*noise norm');
fprintf('PASS: gradient at x_true matches A^T * noise level\n');

%% 6. Finite-difference gradient check (verify grad formula is correct)
fprintf('\n--- Finite-difference gradient check ---\n');
x_test = 0.5 * x_true + 0.1 * randn(Nx*Ny, 1);  % random test point
x_test = max(0, x_test);
g_analytic = gradf(x_test);

eps_fd = 1e-5;
n_check = 10;                             % check 10 random components
fd_errs = zeros(n_check, 1);
idx = randperm(Nx*Ny, n_check);
for i = 1:n_check
    e_i = zeros(Nx*Ny, 1); e_i(idx(i)) = 1;
    fd_errs(i) = abs( (f(x_test + eps_fd*e_i) - f(x_test - eps_fd*e_i)) / (2*eps_fd) ...
                      - g_analytic(idx(i)) );
end
max_fd_err = max(fd_errs);
fprintf('Max finite-difference error (10 components): %.2e\n', max_fd_err);
assert(max_fd_err < 1e-4, 'FAIL: gradient formula has significant error');
fprintf('PASS: gradient formula is correct\n');

%% 7. Lipschitz constant estimate (power iteration, ~20 steps)
fprintf('\n--- Lipschitz constant (largest eigenvalue of A^T A) ---\n');
v = randn(Nx*Ny, 1); v = v / norm(v);
for i = 1:20
    Av   = A * v;
    AtAv = A' * Av;
    lam  = norm(AtAv);
    v    = AtAv / lam;
end
L_f = lam;
fprintf('L_f = %.4f   (used as step size: 1/L_f = %.6f)\n', L_f, 1/L_f);

%% 8. Plot
figure('Name','Piece 1: Forward Model','Position',[100 100 1000 350]);

subplot(1,3,1);
imagesc(WGT); axis image; colorbar;
title(sprintf('Phantom (%dx%d)', Nx, Ny));

subplot(1,3,2);
imagesc(S_img); axis image; colorbar;
xlabel('Detector bin \tau'); ylabel('Angle \theta');
title(sprintf('Sinogram (%d angles, %d bins)', NTheta, NTau));

subplot(1,3,3);
imagesc(reshape(grad_at_true, Ny, Nx)); axis image; colorbar;
title('grad f(x_{true})   [should look like noise]');

fprintf('\n=== Piece 1 PASSED ===\n');
fprintf('Ready to build Piece 2: Restriction/Prolongation operators.\n');
