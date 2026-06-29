function [x, history] = fista(A, s, lambda, L_f, x0, max_iter, tol)
% fista  FISTA solver for tomography: min 0.5*||A*x-s||^2 + lambda*||x||_1
%
% Inputs:
%   A        - forward operator (sparse matrix, m x n)
%   s        - measured sinogram vector (m x 1)
%   lambda   - L1 regularisation weight
%   L_f      - Lipschitz constant of grad_f = ||A||^2  (use estimate from Piece 1)
%   x0       - initial guess (n x 1)
%   max_iter - maximum iterations
%   tol      - stop when ||x_k - x_{k-1}|| / ||x_{k-1}|| < tol
%
% Outputs:
%   x        - reconstructed image vector
%   history  - struct with fields: obj, grad_norm, rel_change, iter

n = numel(x0);

x_k   = x0;
y_k   = x0;
t_k   = 1;

history.obj        = zeros(max_iter, 1);
history.grad_norm  = zeros(max_iter, 1);
history.rel_change = zeros(max_iter, 1);
history.iter       = (1:max_iter)';

step = 1 / L_f;

for k = 1:max_iter
    x_prev = x_k;

    % Gradient step from extrapolated point y_k
    grad = A' * (A * y_k - s);
    x_k  = soft_thresh(y_k - step * grad, step * lambda);

    % Nesterov momentum
    t_next = 0.5 * (1 + sqrt(1 + 4 * t_k^2));
    y_k    = x_k + ((t_k - 1) / t_next) * (x_k - x_prev);
    t_k    = t_next;

    % Record
    history.obj(k)        = 0.5 * norm(A*x_k - s)^2 + lambda * norm(x_k, 1);
    history.grad_norm(k)  = norm(grad);
    history.rel_change(k) = norm(x_k - x_prev) / max(norm(x_prev), 1e-10);

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
% Element-wise soft thresholding (prox of lambda*||.||_1)
x = sign(x) .* max(abs(x) - thresh, 0);
end
