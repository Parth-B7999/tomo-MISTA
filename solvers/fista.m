function [x, history] = fista(A, s, lambda, L_f, x0, max_iter, tol)
% fista  FISTA solver for L1-regularised tomography reconstruction.
%
% PROBLEM SOLVED:
%   min_{x}  F(x)  =  f(x) + g(x)
%
%   where  f(x) = 0.5 * ||A*x - s||^2   (smooth, differentiable data fidelity)
%          g(x) = lambda * ||x||_1       (non-smooth L1 regulariser)
%
%   A  is the tomographic forward operator  (m x n sparse matrix)
%   s  is the measured sinogram vector      (m x 1)
%   x  is the image to reconstruct          (n x 1)
%
% ALGORITHM  (Beck & Teboulle 2009, "FISTA"):
%   FISTA is proximal gradient descent accelerated by Nesterov momentum.
%   Each iteration does three things:
%
%     1. EXTRAPOLATION (momentum step):
%           y_k = x_k + ((t_{k-1} - 1) / t_k) * (x_k - x_{k-1})
%        The Nesterov sequence t_k = 0.5*(1 + sqrt(1 + 4*t_{k-1}^2))
%        gives the optimal O(1/k^2) convergence rate for smooth problems.
%
%     2. GRADIENT STEP at the extrapolated point y_k:
%           u = y_k - (1/L_f) * A'*(A*y_k - s)
%        Step size 1/L_f is guaranteed safe when L_f >= ||A'A|| (Lipschitz
%        constant of grad_f). Use the estimate from power iteration.
%
%     3. PROXIMAL STEP (soft thresholding for L1):
%           x_{k+1} = sign(u) .* max(|u| - lambda/L_f, 0)
%        This is the closed-form solution of
%           prox_{(lambda/L_f)*||.||_1}(u) = argmin_z 0.5*||z-u||^2 + (lambda/L_f)*||z||_1
%
% WHY L_f MATTERS:
%   Step size = 1/L_f. If L_f is underestimated the algorithm diverges.
%   If overestimated it converges but more slowly. Use power iteration
%   (see test_01_forward_model.m) to get a reliable upper bound.
%
% INPUTS:
%   A        - forward projection matrix  (m x n, sparse)
%   s        - sinogram measurement vector  (m x 1)
%   lambda   - L1 regularisation weight (larger = sparser reconstruction)
%   L_f      - Lipschitz constant of grad_f, i.e. largest eigenvalue of A'*A
%   x0       - initial guess (n x 1); zeros(n,1) is fine
%   max_iter - maximum number of iterations
%   tol      - relative-change stopping criterion: stops when
%              ||x_k - x_{k-1}|| / ||x_{k-1}|| < tol  (after 5 iterations)
%
% OUTPUTS:
%   x        - reconstructed image vector (n x 1)
%   history  - struct with per-iteration diagnostics:
%                .obj        F(x_k) = f(x_k) + g(x_k)
%                .grad_norm  ||A'*(A*y_k - s)||  (gradient norm at y_k)
%                .rel_change ||x_k - x_{k-1}|| / ||x_{k-1}||
%                .iter       iteration indices (truncated if early stop)

    step = 1 / L_f;   % gradient step size; safe when L_f >= ||A'A||

    % State variables
    x_k  = x0;
    y_k  = x0;   % extrapolated point (starts equal to x0)
    t_k  = 1;    % Nesterov sequence initialised to 1

    % Pre-allocate history arrays
    history.obj        = zeros(max_iter, 1);
    history.grad_norm  = zeros(max_iter, 1);
    history.rel_change = zeros(max_iter, 1);
    history.iter       = (1:max_iter)';

    for k = 1:max_iter
        x_prev = x_k;   % save previous iterate for momentum and convergence check

        % --- Step 1: Gradient of f at the extrapolated point y_k ---
        % grad_f(y) = A' * (A*y - s)
        residual = A * y_k - s;          % sinogram residual  (m x 1)
        grad     = A' * residual;        % back-projected gradient  (n x 1)

        % --- Step 2 + 3: Proximal gradient step (soft thresholding) ---
        % Combines the gradient step and the L1 proximal operator in one shot.
        % u = y_k - step * grad_f(y_k)
        % x_{k+1} = prox_{step*lambda*||.||_1}(u) = soft_thresh(u, step*lambda)
        x_k = soft_thresh(y_k - step * grad, step * lambda);

        % --- Nesterov momentum update ---
        % Computes new t_{k+1} and builds the extrapolated point y_{k+1}
        t_next = 0.5 * (1 + sqrt(1 + 4 * t_k^2));
        y_k    = x_k + ((t_k - 1) / t_next) * (x_k - x_prev);
        t_k    = t_next;

        % --- Record diagnostics ---
        history.obj(k)        = 0.5 * norm(A*x_k - s)^2 + lambda * norm(x_k, 1);
        history.grad_norm(k)  = norm(grad);
        history.rel_change(k) = norm(x_k - x_prev) / max(norm(x_prev), 1e-10);

        % --- Stopping criterion (skip first 5 iterations to let momentum build) ---
        if history.rel_change(k) < tol && k > 5
            history.obj        = history.obj(1:k);
            history.grad_norm  = history.grad_norm(1:k);
            history.rel_change = history.rel_change(1:k);
            history.iter       = history.iter(1:k);
            break;
        end
    end

    x = x_k;
end


function x = soft_thresh(x, thresh)
% soft_thresh  Element-wise soft thresholding.
%
% This is the closed-form proximal operator of thresh * ||.||_1:
%   prox(u)_i = sign(u_i) * max(|u_i| - thresh, 0)
%
% Values with |u_i| <= thresh are set to zero (sparsity).
% Values with |u_i|  > thresh are shrunk toward zero by thresh.
    x = sign(x) .* max(abs(x) - thresh, 0);
end
