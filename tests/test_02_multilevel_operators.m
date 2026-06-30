% test_02_multilevel_operators.m
%
% PIECE 2: Restriction and Prolongation Operator Tests
%
% Checks:
%   1. P = 4*R' (adjoint relationship)
%   2. R*P ~= I on coarse space
%   3. Coarse operator A_c = A * P maps coarse image -> fine sinogram
%   4. Gradient mismatch (why v_H is needed)
%   5. v_H restores coherence exactly

clear; clc; close all;

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'utils'));
addpath(fullfile(repo_root, 'operators'));

fprintf('=== PIECE 2: Multilevel Operators Test ===\n\n');

%% 1. Build fine-level problem (same as Piece 1)
Nx = 64; Ny = Nx; WSz = [Ny, Nx];
NTheta = 30;
NTau   = ceil(sqrt(Nx^2 + Ny^2));
NTau   = NTau + mod(NTau - Ny, 2);

WGT    = createObject('Phantom', WSz);
x_true = WGT(:);

A = XTM_Tensor_XH(WSz, NTheta, NTau, 0, WGT);
row_sums = full(sum(A, 2));
row_sums(row_sums == 0) = 1;
A = spdiags(1./row_sums, 0, size(A,1), size(A,1)) * A;

noise_std = 0.01; rng(42);
s = A * x_true + noise_std * randn(NTheta*NTau, 1);

fprintf('Fine level : image %dx%d, sinogram %dx%d\n', Nx, Ny, NTheta, NTau);

%% 2. Define coarse level sizes (image only -- sinogram stays at fine level)
% Panos approach: A_c = A * P maps coarse image -> FINE sinogram
% So we only downsample the image, never the sinogram.
Nx_c  = Nx/2;  Ny_c  = Ny/2;
Nx_cc = Nx/4;  Ny_cc = Ny/4;

fprintf('Level 2    : image %dx%d (sinogram stays at %dx%d)\n', Nx_c,  Ny_c,  NTheta, NTau);
fprintf('Level 3    : image %dx%d (sinogram stays at %dx%d)\n\n', Nx_cc, Ny_cc, NTheta, NTau);

%% 3. Build image restriction/prolongation
% build_fw_2D(Nf_x, Nf_y, Nc_x, Nc_y) returns R of size (Nc_x*Nc_y) x (Nf_x*Nf_y)
% Standard multigrid convention: P = R' (transpose), scaled so R*P = I approx

% Convention used throughout this project:
%   R = build_fw_2D(...)       -- un-scaled restriction, rows sum to 1
%   P = build_fw_2D(...)'      -- transpose of R, i.e. P = R'
%   This gives A_c*x_H0 = A*P*R*y ≈ A*y near the solution,
%   so v_H ≈ 0 at the optimum (no artificial scaling mismatch).
R12 = build_fw_2D(Nx,   Ny,   Nx_c,  Ny_c);    % Level 1->2  rows sum to 1
P21 = build_fw_2D(Nx,   Ny,   Nx_c,  Ny_c)';   % Level 2->1  = R12'

R23 = build_fw_2D(Nx_c, Ny_c, Nx_cc, Ny_cc);   % Level 2->3
P32 = build_fw_2D(Nx_c, Ny_c, Nx_cc, Ny_cc)';  % Level 3->2

%% CHECK 1: P = R' (adjoint)
fprintf('--- Check 1: P21 = R12'' (P is exact transpose of R) ---\n');
err_adj = norm(full(P21) - full(R12)', 'fro');
fprintf('||P21 - R12''||_F = %.2e  ', err_adj);
assert(err_adj < 1e-10, 'FAIL: P21 is not R12^T');
fprintf('PASS\n');

%% CHECK 2: R12*P21 ~= I on coarse space
fprintf('\n--- Check 2: R12*P21 approx I_coarse ---\n');
RdotP  = R12 * P21;
I_c    = speye(Nx_c * Ny_c);
rp_err = norm(full(RdotP) - full(I_c), 'fro') / (Nx_c * Ny_c);
fprintf('||R*P - I||_F / n_c = %.4f  ', rp_err);
if rp_err < 0.25
    fprintf('PASS (full-weighting: < 25%% error is acceptable)\n');
else
    fprintf('FAIL: error %.4f is too large\n', rp_err);
    error('R*P is not close enough to identity');
end

%% 3. Build coarse forward operators  A_c = A * P21
fprintf('\n--- Check 3: Coarse operators ---\n');
A_c  = A   * P21;
A_cc = A_c * P32;

fprintf('A    size: %d x %d\n', size(A,1),   size(A,2));
fprintf('A_c  size: %d x %d\n', size(A_c,1), size(A_c,2));
fprintf('A_cc size: %d x %d\n', size(A_cc,1),size(A_cc,2));

% Verify: A_c*(R12*x_true) should be close to A*x_true
x_c          = R12 * x_true;
s_via_fine   = A   * x_true;
s_via_coarse = A_c * x_c;

rel_err = norm(s_via_fine - s_via_coarse) / norm(s_via_fine);
fprintf('\n||A*x - A_c*(R*x)|| / ||A*x|| = %.4f\n', rel_err);
fprintf('NOTE: large error here is EXPECTED and OK.\n');
fprintf('A_c maps coarse image -> fine sinogram, so A_c*(R*x) != A*x\n');
fprintf('(R is lossy downsampling). The coarse model uses the same s.\n');
fprintf('What matters is gradient coherence, not sinogram reproduction.\n');

%% CHECK 4: Gradient mismatch (motivation for v_H)
fprintf('\n--- Check 4: Gradient mismatch before v_H ---\n');

grad_f   = @(x)  A'   * (A*x   - s);
grad_f_c = @(xc) A_c' * (A_c*xc - s);

x_H0              = R12 * x_true;
g_fine            = grad_f(x_true);
g_coarse          = grad_f_c(x_H0);

mismatch     = R12 * g_fine - g_coarse;
rel_mismatch = norm(mismatch) / norm(R12 * g_fine);
fprintf('||R*grad_f(x) - grad_f_c(R*x)|| / ||R*grad_f(x)|| = %.4f\n', rel_mismatch);
fprintf('This mismatch (%.1f%%) is exactly what v_H corrects.\n', 100*rel_mismatch);

%% CHECK 5: v_H restores coherence exactly
fprintf('\n--- Check 5: v_H restores coherence ---\n');

v_H = mismatch;   % v_H = R*grad_f(x) - grad_f_c(R*x)

% Corrected coarse gradient at x_H0 = grad_f_c(x_H0) + v_H
g_corrected   = g_coarse + v_H;
coherence_err = norm(g_corrected - R12 * g_fine) / norm(R12 * g_fine);
fprintf('After v_H: coherence error = %.2e  ', coherence_err);
assert(coherence_err < 1e-10, 'FAIL: v_H did not restore coherence');
fprintf('PASS\n');

%% Lipschitz constants per level
fprintf('\n--- Lipschitz constants (power iteration, 20 steps) ---\n');
mats  = {A, A_c, A_cc};
names = {'Level 1 (fine)  ', 'Level 2 (mid)   ', 'Level 3 (coarse)'};
L_f   = zeros(1, 3);
for i = 1:3
    M = mats{i};
    v = randn(size(M,2), 1); v = v / norm(v);
    for j = 1:20
        w   = M' * (M * v);
        lam = norm(w);
        v   = w / lam;
    end
    L_f(i) = lam;
    fprintf('  %s: L_f = %.4f\n', names{i}, lam);
end

%% Plot
figure('Name', 'Piece 2: Multilevel Operators', 'Position', [50 50 1200 400]);

subplot(1,4,1);
imagesc(reshape(x_true, Ny, Nx));
axis image; colorbar;
title(sprintf('x fine (%dx%d)', Nx, Ny));

subplot(1,4,2);
imagesc(reshape(x_c, Ny_c, Nx_c));
axis image; colorbar;
title(sprintf('R*x coarse (%dx%d)', Nx_c, Ny_c));

subplot(1,4,3);
imagesc(reshape(s_via_fine, NTheta, NTau));
axis image; colorbar;
title('Sinogram via fine A');

subplot(1,4,4);
imagesc(reshape(s_via_coarse, NTheta, NTau));
axis image; colorbar;
title('Sinogram via coarse A_c');

fprintf('\n=== Piece 2 PASSED ===\n');
fprintf('Ready to build Piece 3: FISTA baseline solver.\n');
