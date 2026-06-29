function [x, history] = mrfista(A, A_c, R, P, s, lambda, L_f, L_fc, x0, max_iter, tol, params)
% mrfista  2-level Multilevel FISTA (MR-FISTA) for L1-regularised tomography.
%
% PROBLEM SOLVED (same as fista.m):
%   min_{x}  F(x) = f(x) + g(x)
%           f(x) = 0.5 * ||A*x - s||^2      (smooth data fidelity, fine level)
%           g(x) = lambda * ||x||_1          (L1 regulariser)
%
% IDEA (Parpas 2017, MISTA):
%   Most of FISTA's iterations are standard proximal gradient steps.
%   When the gradient is large on the coarse grid (i.e. the coarse model can
%   give useful information), we replace one fine-level step with a cheaper
%   coarse-level solve that provides a better search direction.
%
%   The coarse operator is A_c = A * P  (fine operator composed with prolongation).
%   A_c maps a COARSE image to the FINE sinogram, so the measurement s stays
%   at fine resolution throughout.  Only the image is downsampled.
%
% ALGORITHM (one outer iteration):
%
%   1. FISTA EXTRAPOLATION
%         y_k = x_k + ((t_{k-1}-1)/t_k) * (x_k - x_{k-1})
%      Standard Nesterov momentum with sequence t_k -> O(1/k^2) rate.
%
%   2. GRADIENT AT y_k
%         grad_f(y_k) = A'*(A*y_k - s)
%
%   3. SMOOTHED GRADIENT (for decision and v_H only -- NOT used in the fine step)
%         grad_g_mu(y) = lambda * y ./ sqrt(mu^2 + y.^2)
%         grad_F_mu(y) = grad_f(y) + grad_g_mu(y)
%      mu is a small constant that makes ||x||_1 differentiable.
%      This gives a smooth surrogate of the full gradient, needed because
%      the coarse model must be differentiable (MISTA Assumption 1).
%
%   4. DECISION: coarse step or gradient step?
%         cond1: ||R * grad_F_mu(y_k)|| > kappa * ||grad_F_mu(y_k)||
%                --> gradient is not negligible on the coarse grid
%                --> coarse model carries useful curvature information
%         cond2: ||y_k - x_tilde|| > theta * ||x_tilde||  OR  q < Kd
%                --> not too close to the last correction point, OR
%                    haven't accumulated enough gradient steps yet
%      do_coarse = cond1 AND cond2
%
%   5a. COARSE CORRECTION STEP (if do_coarse):
%
%       Restrict current point to coarse grid:
%           x_H0 = R * y_k                       (coarse initial guess)
%
%       Compute the first-order coherence term v_H:
%           grad_F_mu_c(x_H0) = A_c'*(A_c*x_H0 - s) + lambda*x_H0./sqrt(mu^2+x_H0.^2)
%           v_H = R * grad_F_mu(y_k) - grad_F_mu_c(x_H0)
%
%       v_H ensures that the coarse and fine gradients agree at x_H0:
%           grad_F_c_corrected(x_H0) = grad_F_mu_c(x_H0) + v_H
%                                    = R * grad_F_mu(y_k)    [exactly]
%       (Verified in test_02: coherence error = 7.5e-15)
%
%       Solve the corrected coarse problem (NH iterations of proximal gradient):
%           min_{x_H}  0.5*||A_c*x_H - s||^2 + <v_H, x_H> + lambda*||x_H||_1
%           grad of smooth part = A_c'*(A_c*x_H - s) + v_H
%           prox of nonsmooth part = soft_thresh(., lambda/L_fc)
%
%       Coarse correction direction (prolonged to fine grid):
%           d = P * (x_H_star - x_H0)
%           (direction from restricted starting point toward coarse solution)
%
%       Armijo backtracking line search along d:
%           Find s_k > 0 such that F(y_k + s_k*d) <= F(y_k) + c*s_k*<grad_F_mu(y_k),d>
%           If d is not a descent direction (inner product >= 0), fall back to grad step.
%
%       Update:  x_{k+1} = y_k + s_k * d
%
%   5b. GRADIENT STEP (if NOT do_coarse):
%       Standard FISTA proximal gradient step from y_k:
%           x_{k+1} = soft_thresh(y_k - (1/L_f)*grad_f(y_k), lambda/L_f)
%
%   6. NESTEROV UPDATE
%           t_{k+1} = 0.5*(1 + sqrt(1 + 4*t_k^2))
%           x_{k-1} = x_k,  x_k = x_{k+1}
%
% INPUTS:
%   A        - fine forward operator  (m x n, sparse)
%   A_c      - coarse forward operator  (m x n_c, where n_c = n/4 for 2x downsampling)
%              A_c = A * P  (built in setup/test_02)
%   R        - restriction operator  (n_c x n, sparse)
%              restricts fine image to coarse image
%   P        - prolongation operator  (n x n_c, sparse)
%              P = 4*R'  (bilinear interpolation, verified in test_02)
%   s        - measured sinogram vector  (m x 1)
%   lambda   - L1 regularisation weight
%   L_f      - Lipschitz constant of grad_f at fine level  (= largest eigenvalue of A'*A)
%   L_fc     - Lipschitz constant of grad_f at coarse level (= largest eigenvalue of A_c'*A_c)
%   x0       - initial guess  (n x 1)
%   max_iter - maximum outer iterations
%   tol      - stopping criterion: relative change ||x_k - x_{k-1}||/||x_{k-1}|| < tol
%   params   - struct with fields:
%       .kappa  decision threshold for coarse step  (default 0.3)
%       .theta  proximity threshold                  (default 0.5)
%       .Kd     max consecutive grad steps           (default 5)
%       .mu     smoothing parameter for ||x||_1     (default 1e-4)
%       .NH     number of coarse iterations          (default 10)
%       .c      Armijo line search parameter         (default 1e-4)
%       .tau    backtracking shrinkage factor        (default 0.5)
%
% OUTPUTS:
%   x        - reconstructed image vector  (n x 1)
%   history  - struct with per-iteration diagnostics:
%       .obj           F(x_k) = f(x_k) + g(x_k)
%       .rel_change    ||x_k - x_{k-1}|| / ||x_{k-1}||
%       .step_type     string: 'coarse' or 'grad'
%       .coarse_s_k    line search step sizes (NaN for gradient steps)
%       .iter          iteration indices

    % --- Unpack parameters (with defaults) ---
    kappa = get_param(params, 'kappa', 0.3);
    theta = get_param(params, 'theta', 0.5);
    Kd    = get_param(params, 'Kd',    5);
    mu    = get_param(params, 'mu',    1e-4);
    NH    = get_param(params, 'NH',    10);
    c     = get_param(params, 'c',     1e-4);
    tau   = get_param(params, 'tau',   0.5);

    step_f  = 1 / L_f;    % fine-level step size
    step_fc = 1 / L_fc;   % coarse-level step size

    % --- Initialise state ---
    x_k    = x0;
    x_prev = x0;
    t_k    = 1;
    t_prev = 1;

    x_tilde = x0;   % last point where a coarse correction was accepted
    q = 0;          % consecutive gradient steps since last coarse

    % --- Pre-allocate history ---
    history.obj        = zeros(max_iter, 1);
    history.rel_change = zeros(max_iter, 1);
    history.step_type  = cell(max_iter, 1);
    history.coarse_s_k = nan(max_iter, 1);
    history.iter       = (1:max_iter)';

    for k = 1:max_iter
        x_prev_k = x_k;   % save for rel_change and Nesterov

        % ----------------------------------------------------------------
        % STEP 1: FISTA extrapolation (Nesterov momentum)
        % ----------------------------------------------------------------
        % Combines x_k with the previous iterate using the Nesterov coefficient.
        % This gives the extrapolated point y_k from which we compute the gradient.
        y_k = x_k + ((t_prev - 1) / t_k) * (x_k - x_prev);

        % ----------------------------------------------------------------
        % STEP 2: Gradient of the smooth part f at y_k
        % ----------------------------------------------------------------
        residual_y  = A * y_k - s;            % fine sinogram residual  (m x 1)
        grad_f_y    = A' * residual_y;        % backprojected gradient  (n x 1)

        % ----------------------------------------------------------------
        % STEP 3: Smoothed full gradient at y_k (for decision and v_H)
        % ----------------------------------------------------------------
        % Moreau envelope approximation of grad ||y||_1:
        %   grad_g_mu(y)_i = lambda * y_i / sqrt(mu^2 + y_i^2)
        % As mu -> 0 this approaches lambda * sign(y_i).
        % We use mu > 0 so the coarse model is differentiable.
        grad_g_mu_y   = lambda * y_k ./ sqrt(mu^2 + y_k.^2);
        grad_F_mu_y   = grad_f_y + grad_g_mu_y;

        % ----------------------------------------------------------------
        % STEP 4: Decision -- coarse or gradient step?
        % ----------------------------------------------------------------
        % We use two different restrictions of the gradient:
        %
        %   grad_restricted_scaled = R * grad    (uses normalised R = (1/4)*R_unnorm)
        %     -> needed for v_H: must match the scale of x_H0 = R * y_k
        %
        %   grad_restricted_unnorm = P' * grad   (un-normalised, = R_unnorm * grad)
        %     -> used for the DECISION ONLY, to get a scale-invariant comparison
        %     -> norm(P' * g) ≈ 0.5 * norm(g) for typical gradient vectors,
        %        which is meaningfully compared to kappa (e.g. kappa = 0.3)
        %
        % If we used norm(R * g) in the decision, the (1/4) scaling would make
        % norm(R*g) ≈ 0.125 * norm(g) -- always below kappa = 0.3 and coarse
        % steps would never trigger.
        grad_restricted_scaled = R  * grad_F_mu_y;   % (1/4)-scaled, for v_H
        grad_restricted_unnorm = P' * grad_F_mu_y;   % un-normalised, for decision

        % cond1: coarse gradient is not negligible relative to fine gradient
        cond1 = norm(grad_restricted_unnorm) > kappa * norm(grad_F_mu_y);

        % cond2: either we haven't done too many consecutive grad steps,
        %        or we have moved far enough from the last correction point
        cond2 = (q < Kd) || (norm(y_k - x_tilde) > theta * norm(x_tilde));

        do_coarse = cond1 && cond2;

        % ----------------------------------------------------------------
        % STEP 5a: COARSE CORRECTION STEP
        % ----------------------------------------------------------------
        if do_coarse

            % -- Restrict current point to coarse grid --
            x_H0 = R * y_k;    % coarse starting point  (n_c x 1)

            % -- Compute coherence correction v_H --
            % Gradient of smooth coarse objective at x_H0 (without v_H):
            %   A_c'*(A_c*x_H0 - s) + grad_g_mu_c(x_H0)
            residual_H0      = A_c * x_H0 - s;
            grad_f_c_H0      = A_c' * residual_H0;
            grad_g_mu_c_H0   = lambda * x_H0 ./ sqrt(mu^2 + x_H0.^2);
            grad_F_mu_c_H0   = grad_f_c_H0 + grad_g_mu_c_H0;

            % v_H forces the coarse gradient to match the restricted fine gradient:
            %   grad_F_mu_c(x_H0) + v_H = R * grad_F_mu(y_k)
            % Uses grad_restricted_SCALED (= R * grad) so units match:
            % both sides are in the (1/4)-normalised coarse space.
            v_H = grad_restricted_scaled - grad_F_mu_c_H0;

            % -- Solve corrected coarse problem (NH proximal gradient steps) --
            % Objective: 0.5*||A_c*x_H - s||^2 + <v_H, x_H> + lambda*||x_H||_1
            % Smooth gradient: A_c'*(A_c*x_H - s) + v_H
            x_H = x_H0;
            for j = 1:NH
                grad_c = A_c' * (A_c * x_H - s) + v_H;
                x_H    = soft_thresh(x_H - step_fc * grad_c, step_fc * lambda);
            end
            x_H_star = x_H;

            % -- Coarse correction direction (prolonged to fine grid) --
            % d points from the restricted starting point toward the coarse solution.
            d = P * (x_H_star - x_H0);

            % -- Check descent direction --
            % We need <grad_F_mu(y_k), d> < 0 for Armijo to make sense.
            descent_term = grad_F_mu_y' * d;

            if descent_term >= 0
                % Coarse direction is not a descent direction for the fine objective.
                % This can happen when the coarse model is a poor surrogate.
                % Fall back to a standard gradient step.
                x_next = soft_thresh(y_k - step_f * grad_f_y, step_f * lambda);
                q      = q + 1;
                s_k    = 0;
                history.step_type{k} = 'grad_fallback';
            else
                % -- Armijo backtracking line search along d --
                % Find the largest step s_k satisfying the sufficient decrease condition:
                %   F(y_k + s_k*d) <= F(y_k) + c * s_k * <grad_F_mu(y_k), d>
                F_y = 0.5 * norm(residual_y)^2 + lambda * norm(y_k, 1);
                s_k = 1.0;
                x_next = y_k;   % default if line search fails immediately
                for ls = 1:30
                    x_cand = y_k + s_k * d;
                    F_cand = 0.5 * norm(A * x_cand - s)^2 + lambda * norm(x_cand, 1);
                    if F_cand <= F_y + c * s_k * descent_term
                        x_next = x_cand;
                        break;
                    end
                    s_k = s_k * tau;
                    if s_k < 1e-8
                        % Line search failed: step is too small, fall back to grad step
                        x_next = soft_thresh(y_k - step_f * grad_f_y, step_f * lambda);
                        s_k    = 0;
                        break;
                    end
                end

                if s_k > 0
                    q = 0;          % reset consecutive gradient step counter
                    x_tilde = y_k;  % record last accepted coarse correction point
                    history.step_type{k} = 'coarse';
                else
                    q = q + 1;
                    history.step_type{k} = 'grad_fallback';
                end
            end

            history.coarse_s_k(k) = s_k;

        % ----------------------------------------------------------------
        % STEP 5b: STANDARD GRADIENT STEP
        % ----------------------------------------------------------------
        else
            % Standard FISTA proximal gradient step from y_k.
            % Same as fista.m but we already have grad_f_y from Step 2.
            x_next = soft_thresh(y_k - step_f * grad_f_y, step_f * lambda);
            q      = q + 1;
            history.step_type{k} = 'grad';
        end

        % ----------------------------------------------------------------
        % STEP 6: Nesterov sequence update
        % ----------------------------------------------------------------
        t_next = 0.5 * (1 + sqrt(1 + 4 * t_k^2));
        x_prev = x_k;     % shift: previous becomes the one before
        x_k    = x_next;  % current becomes the new iterate
        t_prev = t_k;
        t_k    = t_next;

        % ----------------------------------------------------------------
        % Record diagnostics and check stopping criterion
        % ----------------------------------------------------------------
        history.obj(k)        = 0.5 * norm(A*x_k - s)^2 + lambda * norm(x_k, 1);
        history.rel_change(k) = norm(x_k - x_prev_k) / max(norm(x_prev_k), 1e-10);
        history.iter(k)       = k;

        if history.rel_change(k) < tol && k > 5
            history.obj        = history.obj(1:k);
            history.rel_change = history.rel_change(1:k);
            history.step_type  = history.step_type(1:k);
            history.coarse_s_k = history.coarse_s_k(1:k);
            history.iter       = history.iter(1:k);
            break;
        end
    end

    x = x_k;
end


% -------------------------------------------------------------------------
function x = soft_thresh(x, thresh)
% soft_thresh  Element-wise soft thresholding (prox of thresh * ||.||_1).
%   Values with |x_i| <= thresh are zeroed (sparsity).
%   Values with |x_i| >  thresh are shrunk toward zero by thresh.
    x = sign(x) .* max(abs(x) - thresh, 0);
end


% -------------------------------------------------------------------------
function val = get_param(params, name, default)
% get_param  Read a field from params struct, returning default if absent.
    if isfield(params, name)
        val = params.(name);
    else
        val = default;
    end
end
