% test_05_benchmark.m
%
% PIECE 5: Proper benchmark -- two experiments that reveal when MR-FISTA helps.
%
% =========================================================================
% PART A -- OPTION B: Limited-Angle CT  (why FISTA can be slow)
% =========================================================================
%   With very few projection angles the system A is severely rank-deficient.
%   FISTA needs many iterations because the gradient is a poor predictor of
%   the true descent direction (the condition number is large).
%   We show MR-FISTA (one-sided) outperforms FISTA in this regime.
%
% =========================================================================
% PART B -- OPTION A: Two-Sided Restriction  (the proper multilevel fix)
% =========================================================================
%   With one-sided coarse model (A_c = A*P, s_c = s) the coarse problem
%   is overdetermined and v_H is enormous -- the coarse correction adds
%   little beyond a gradient step.
%
%   With two-sided coarse model (A_c = Rs*A*P, s_c = Rs*s) the coarse
%   problem preserves the underdetermined character of the fine problem
%   and v_H is small -- the coarse solution is a genuinely better direction.
%
%   Comparison: FISTA vs MR-FISTA(one-sided) vs MR-FISTA(two-sided).

clear; clc; close all;

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'utils'));
addpath(fullfile(repo_root, 'operators'));
addpath(fullfile(repo_root, 'solvers'));

%% ========================================================================
%% PART A: Limited-Angle CT
%% ========================================================================
fprintf('==========================================\n');
fprintf('PART A: Limited-Angle CT (NTheta = 10)\n');
fprintf('==========================================\n\n');

Nx_A = 64; Ny_A = Nx_A; WSz_A = [Ny_A, Nx_A];
NTheta_A = 10;   % very few angles -- severely underdetermined AND rank-deficient
NTau_A   = ceil(sqrt(Nx_A^2 + Ny_A^2));
NTau_A   = NTau_A + mod(NTau_A - Ny_A, 2);

WGT_A  = createObject('Phantom', WSz_A);
x_true_A = WGT_A(:);

fprintf('Image     : %dx%d = %d unknowns\n', Nx_A, Ny_A, Nx_A*Ny_A);
fprintf('Sinogram  : %d angles x %d bins = %d measurements\n', ...
    NTheta_A, NTau_A, NTheta_A*NTau_A);
fprintf('System    : %.1f%% determined (< 100%% = underdetermined)\n\n', ...
    100 * NTheta_A*NTau_A / (Nx_A*Ny_A));

% Build and normalise forward operator
A_A = XTM_Tensor_XH(WSz_A, NTheta_A, NTau_A, 0, WGT_A);
rs_A = full(sum(A_A, 2)); rs_A(rs_A == 0) = 1;
A_A = spdiags(1./rs_A, 0, size(A_A,1), size(A_A,1)) * A_A;

noise_std = 0.01; rng(42);
s_A = A_A * x_true_A + noise_std * randn(NTheta_A*NTau_A, 1);

% Coarse level (2x downsampling of image only -- one-sided)
Nx_Ac = Nx_A/2; Ny_Ac = Ny_A/2;
R_A = build_fw_2D(Nx_A, Ny_A, Nx_Ac, Ny_Ac);   % un-scaled, rows sum to 1
P_A = build_fw_2D(Nx_A, Ny_A, Nx_Ac, Ny_Ac)';  % = R_A'
Ac_A  = A_A * P_A;   % one-sided coarse operator

fprintf('Coarse (one-sided) : A_c is %dx%d\n', size(Ac_A,1), size(Ac_A,2));
fprintf('  Coarse problem   : %d measurements, %d unknowns (%.1f%% determined)\n\n', ...
    size(Ac_A,1), size(Ac_A,2), 100*size(Ac_A,1)/size(Ac_A,2));

% Lipschitz constants
v = randn(Nx_A*Ny_A, 1); v = v/norm(v);
for i = 1:20; w = A_A'*(A_A*v); L_fA = norm(w); v = w/L_fA; end

v = randn(Nx_Ac*Ny_Ac, 1); v = v/norm(v);
for i = 1:20; w = Ac_A'*(Ac_A*v); L_fcA = norm(w); v = w/L_fcA; end

fprintf('L_f (fine)  = %.4f\n', L_fA);
fprintf('L_fc(coarse)= %.4f\n\n', L_fcA);

lambda_A   = 1e-3;
max_iter_A = 500;
tol_A      = 1e-9;   % tight: let both run their budget
x0_A       = zeros(Nx_A*Ny_A, 1);

params_A.kappa = 0.45;
params_A.theta = 0.5;
params_A.Kd    = 5;
params_A.mu    = 1e-4;
params_A.NH    = 20;   % more coarse iterations for limited-angle
params_A.c     = 1e-4;
params_A.tau   = 0.5;

fprintf('--- Running FISTA (limited-angle baseline) ---\n');
tic;
[x_fA, hf_A] = fista(A_A, s_A, lambda_A, L_fA, x0_A, max_iter_A, tol_A);
tf_A = toc;
fprintf('  Iters: %d   Time: %.3fs   Final obj: %.4e\n\n', ...
    length(hf_A.iter), tf_A, hf_A.obj(end));

fprintf('--- Running MR-FISTA one-sided (limited-angle) ---\n');
tic;
[x_mA, hm_A] = mrfista(A_A, Ac_A, R_A, P_A, s_A, s_A, ...
    lambda_A, L_fA, L_fcA, x0_A, max_iter_A, tol_A, params_A);
tm_A = toc;
n_coarse_A = sum(strcmp(hm_A.step_type, 'coarse'));
fprintf('  Iters: %d   Time: %.3fs   Final obj: %.4e\n', ...
    length(hm_A.iter), tm_A, hm_A.obj(end));
fprintf('  Coarse: %d   Gradient: %d\n\n', ...
    n_coarse_A, sum(strcmp(hm_A.step_type,'grad')));

% Compare convergence at equal iteration count
n_common_A = min(length(hf_A.obj), length(hm_A.obj));
fprintf('--- Part A results at iter %d ---\n', n_common_A);
fprintf('  FISTA    obj = %.6e\n', hf_A.obj(n_common_A));
fprintf('  MR-FISTA obj = %.6e\n', hm_A.obj(n_common_A));
if hm_A.obj(n_common_A) < hf_A.obj(n_common_A)
    speedup = hf_A.obj(n_common_A) / hm_A.obj(n_common_A);
    fprintf('  MR-FISTA reaches %.2fx lower objective at same iteration count.\n\n', speedup);
else
    fprintf('  No benefit at same iteration count (may appear at equal work budget).\n\n');
end

%% ========================================================================
%% PART B: Two-Sided Restriction
%% ========================================================================
fprintf('==========================================\n');
fprintf('PART B: Two-Sided Restriction Comparison\n');
fprintf('  (same 30-angle problem as Pieces 1-4)  \n');
fprintf('==========================================\n\n');

Nx_B = 64; Ny_B = Nx_B; WSz_B = [Ny_B, Nx_B];
NTheta_B = 30;
NTau_B   = ceil(sqrt(Nx_B^2 + Ny_B^2));
NTau_B   = NTau_B + mod(NTau_B - Ny_B, 2);

WGT_B    = createObject('Phantom', WSz_B);
x_true_B = WGT_B(:);

A_B = XTM_Tensor_XH(WSz_B, NTheta_B, NTau_B, 0, WGT_B);
rs_B = full(sum(A_B, 2)); rs_B(rs_B == 0) = 1;
A_B = spdiags(1./rs_B, 0, size(A_B,1), size(A_B,1)) * A_B;

rng(42);
s_B = A_B * x_true_B + noise_std * randn(NTheta_B*NTau_B, 1);

% --- Coarse level image operators (same for both one-sided and two-sided) ---
Nx_Bc = Nx_B/2; Ny_Bc = Ny_B/2;
NTheta_Bc = NTheta_B/2; NTau_Bc = NTau_B/2;

R_B  = build_fw_2D(Nx_B, Ny_B, Nx_Bc, Ny_Bc);    % un-scaled, rows sum to 1
P_B  = build_fw_2D(Nx_B, Ny_B, Nx_Bc, Ny_Bc)';   % = R_B'            % image prolongation

% --- One-sided coarse operator (A_c = A * P, s_c = s) ---
Ac_B_1sided = A_B * P_B;

% --- Two-sided coarse operator (A_c = Rs * A * P, s_c = Rs * s) ---
% Rs restricts the sinogram: (NTheta*NTau) -> (NTheta_c*NTau_c)
Rs_B = build_fw_2D(NTheta_B, NTau_B, NTheta_Bc, NTau_Bc);
s_Bc = Rs_B * s_B;                                      % restricted sinogram
Ac_B_2sided = restrict_matrix(A_B, NTheta_B, NTau_B, ...
    NTheta_Bc, NTau_Bc, Nx_B, Ny_B, Nx_Bc, Ny_Bc);    % Rs * A * P

fprintf('One-sided coarse A_c : %d x %d  (m >= n_c -- overdetermined)\n', ...
    size(Ac_B_1sided,1), size(Ac_B_1sided,2));
fprintf('Two-sided coarse A_c : %d x %d  (m_c < n_c -- underdetermined like fine)\n\n', ...
    size(Ac_B_2sided,1), size(Ac_B_2sided,2));

% --- Check v_H magnitude for both approaches ---
v = randn(Nx_B*Ny_B, 1); v = v/norm(v);
for i = 1:20; w = A_B'*(A_B*v); L_fB = norm(w); v = w/L_fB; end

v = randn(Nx_Bc*Ny_Bc, 1); v = v/norm(v);
for i = 1:20; w = Ac_B_1sided'*(Ac_B_1sided*v); L_fcB_1s = norm(w); v = w/L_fcB_1s; end

v = randn(Nx_Bc*Ny_Bc, 1); v = v/norm(v);
for i = 1:20; w = Ac_B_2sided'*(Ac_B_2sided*v); L_fcB_2s = norm(w); v = w/L_fcB_2s; end

% Measure v_H at x_true to compare one-sided vs two-sided
x_H0_B   = R_B * x_true_B;
g_fine_B  = A_B' * (A_B * x_true_B - s_B);
Rg_fine_B = R_B * g_fine_B;

g_c_1sided = Ac_B_1sided' * (Ac_B_1sided * x_H0_B - s_B);
g_c_2sided = Ac_B_2sided' * (Ac_B_2sided * x_H0_B - s_Bc);

vH_1sided = Rg_fine_B - g_c_1sided;
vH_2sided = Rg_fine_B - g_c_2sided;

fprintf('--- v_H magnitude at x_true ---\n');
fprintf('  One-sided ||v_H|| / ||R*grad|| = %.2f  (%.0f%%)\n', ...
    norm(vH_1sided)/norm(Rg_fine_B), 100*norm(vH_1sided)/norm(Rg_fine_B));
fprintf('  Two-sided ||v_H|| / ||R*grad|| = %.4f  (%.2f%%)\n\n', ...
    norm(vH_2sided)/norm(Rg_fine_B), 100*norm(vH_2sided)/norm(Rg_fine_B));

lambda_B   = 1e-3;
max_iter_B = 200;
tol_B      = 1e-9;
x0_B       = zeros(Nx_B*Ny_B, 1);

params_B.kappa = 0.45;
params_B.theta = 0.5;
params_B.Kd    = 5;
params_B.mu    = 1e-4;
params_B.NH    = 20;
params_B.c     = 1e-4;
params_B.tau   = 0.5;

fprintf('--- Running FISTA ---\n');
tic;
[x_fB, hf_B] = fista(A_B, s_B, lambda_B, L_fB, x0_B, max_iter_B, tol_B);
tf_B = toc;
fprintf('  Iters: %d   Time: %.3fs   Final obj: %.4e\n\n', ...
    length(hf_B.iter), tf_B, hf_B.obj(end));

fprintf('--- Running MR-FISTA one-sided ---\n');
tic;
[x_m1B, hm1_B] = mrfista(A_B, Ac_B_1sided, R_B, P_B, s_B, s_B, ...
    lambda_B, L_fB, L_fcB_1s, x0_B, max_iter_B, tol_B, params_B);
tm1_B = toc;
nc1 = sum(strcmp(hm1_B.step_type,'coarse'));
fprintf('  Iters: %d   Time: %.3fs   Final obj: %.4e   Coarse: %d\n\n', ...
    length(hm1_B.iter), tm1_B, hm1_B.obj(end), nc1);

fprintf('--- Running MR-FISTA two-sided ---\n');
tic;
[x_m2B, hm2_B] = mrfista(A_B, Ac_B_2sided, R_B, P_B, s_B, s_Bc, ...
    lambda_B, L_fB, L_fcB_2s, x0_B, max_iter_B, tol_B, params_B);
tm2_B = toc;
nc2 = sum(strcmp(hm2_B.step_type,'coarse'));
fprintf('  Iters: %d   Time: %.3fs   Final obj: %.4e   Coarse: %d\n\n', ...
    length(hm2_B.iter), tm2_B, hm2_B.obj(end), nc2);

%% ========================================================================
%% Checks
%% ========================================================================
fprintf('--- Checks ---\n');

% All three objectives must be monotone
hists_B = {hf_B, hm1_B, hm2_B};
names_B = {'FISTA', 'MR-FISTA-1sided', 'MR-FISTA-2sided'};
for i = 1:3
    d = max(diff(hists_B{i}.obj));
    fprintf('Monotone %-20s: max increase = %.2e  ', names_B{i}, d);
    assert(d < 1e-8 * hists_B{i}.obj(1), ['FAIL: ' names_B{i} ' objective not monotone']);
    fprintf('PASS\n');
end

% Two-sided v_H should be much smaller than one-sided
ratio_vH = norm(vH_2sided) / norm(vH_1sided);
fprintf('v_H ratio (2sided/1sided) = %.4f  ', ratio_vH);
assert(ratio_vH < 0.5, 'FAIL: two-sided v_H is not smaller than one-sided');
fprintf('PASS (two-sided v_H is %.1fx smaller)\n', 1/ratio_vH);

% At least some coarse steps accepted in both MR-FISTA runs
assert(nc1 > 0, 'FAIL: no coarse steps in one-sided MR-FISTA');
assert(nc2 > 0, 'FAIL: no coarse steps in two-sided MR-FISTA');
fprintf('Coarse steps: one-sided=%d, two-sided=%d  PASS\n', nc1, nc2);

%% ========================================================================
%% Summary Table
%% ========================================================================
n = min([length(hf_B.obj), length(hm1_B.obj), length(hm2_B.obj)]);
fprintf('\n--- Summary at iteration %d (Part B, 30-angle problem) ---\n', n);
fprintf('  %-22s  obj = %.4e\n', 'FISTA',           hf_B.obj(n));
fprintf('  %-22s  obj = %.4e\n', 'MR-FISTA (1-sided)', hm1_B.obj(n));
fprintf('  %-22s  obj = %.4e\n', 'MR-FISTA (2-sided)', hm2_B.obj(n));

%% ========================================================================
%% Plots
%% ========================================================================

% --- Part A: convergence ---
figure('Name','Piece 5 Part A: Limited-Angle','Position',[50 50 1200 450]);

subplot(1,3,1);
semilogy(hf_A.iter, hf_A.obj, 'b-', 'LineWidth', 1.5); hold on;
semilogy(hm_A.iter, hm_A.obj, 'r-', 'LineWidth', 1.5);
coarse_idx_A = find(strcmp(hm_A.step_type,'coarse'));
if ~isempty(coarse_idx_A)
    semilogy(coarse_idx_A, hm_A.obj(coarse_idx_A), 'r^', ...
        'MarkerFaceColor','r','MarkerSize',4);
end
hold off; grid on;
xlabel('Iteration'); ylabel('Objective');
title(sprintf('Limited-Angle (NTheta=%d): convergence', NTheta_A));
legend('FISTA','MR-FISTA','Coarse steps');

subplot(1,3,2);
imagesc(reshape(x_fA, Ny_A, Nx_A)); axis image; colorbar;
title(sprintf('FISTA (%.3fs)', tf_A));

subplot(1,3,3);
imagesc(reshape(x_mA, Ny_A, Nx_A)); axis image; colorbar;
title(sprintf('MR-FISTA one-sided (%.3fs)', tm_A));

% --- Part B: three-way convergence comparison ---
figure('Name','Piece 5 Part B: Two-Sided Restriction','Position',[50 550 1400 450]);

subplot(1,4,1);
semilogy(hf_B.iter,  hf_B.obj,  'b-',  'LineWidth',1.5); hold on;
semilogy(hm1_B.iter, hm1_B.obj, 'r--', 'LineWidth',1.5);
semilogy(hm2_B.iter, hm2_B.obj, 'g-',  'LineWidth',1.5);
hold off; grid on;
xlabel('Iteration'); ylabel('Objective');
title('Part B: 3-way convergence');
legend('FISTA','MR-FISTA 1-sided','MR-FISTA 2-sided');

subplot(1,4,2);
imagesc(reshape(x_fB, Ny_B, Nx_B)); axis image; colorbar;
title('FISTA');

subplot(1,4,3);
imagesc(reshape(x_m1B, Ny_B, Nx_B)); axis image; colorbar;
title('MR-FISTA 1-sided');

subplot(1,4,4);
imagesc(reshape(x_m2B, Ny_B, Nx_B)); axis image; colorbar;
title('MR-FISTA 2-sided');

fprintf('\n=== Piece 5 DONE ===\n');
