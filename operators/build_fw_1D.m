function R = build_fw_1D(N_fine, N_coarse)
% build_fw_1D  Build a 1D full-weighting restriction operator.
%
% PURPOSE:
%   Restricts a vector of length N_fine to a vector of length N_coarse
%   using a 4-point full-weighting stencil with linear interpolation weights.
%   Works for any ratio N_fine / N_coarse (not limited to powers of 2).
%
% HOW IT WORKS:
%   Each coarse point i is placed at position  pos = (i - 0.5) * scale
%   in fine-grid coordinates, where scale = N_fine / N_coarse.
%
%   The value at the coarse point is a weighted average of the 4 nearest
%   fine-grid neighbours:
%
%     alpha = fractional offset of pos within its fine-grid cell
%
%     fine[left_idx - 1]  weight = (1 - alpha) * 0.25   (left neighbour)
%     fine[left_idx    ]  weight = (1 - alpha) * 0.50   (left of centre)
%     fine[left_idx + 1]  weight =       alpha * 0.50   (right of centre)
%     fine[left_idx + 2]  weight =       alpha * 0.25   (right neighbour)
%
%   Out-of-bounds fine indices are silently dropped (boundary handling).
%   Each row is then re-normalised to sum to 1 so that a constant fine-grid
%   signal maps to the same constant on the coarse grid.
%
% RELATIONSHIP TO PROLONGATION:
%   The prolongation operator P (coarse -> fine) is the transpose: P = R'.
%   Together they satisfy  R * P  ~=  I  (approximate identity on coarse space).
%
% INPUT:
%   N_fine   - number of fine-grid points
%   N_coarse - number of coarse-grid points  (must be <= N_fine)
%
% OUTPUT:
%   R - sparse restriction matrix of size  (N_coarse x N_fine)

    scale = N_fine / N_coarse;   % ratio of grid spacings

    rows = [];
    cols = [];
    vals = [];

    for i = 1:N_coarse
        % Position of coarse point i in fine-grid coordinates
        pos      = (i - 0.5) * scale;
        left_idx = floor(pos);
        alpha    = pos - left_idx;   % fractional offset in [0, 1)

        % 4-point stencil: index and weight pairs
        idx_weights = [ ...
            left_idx - 1,  (1 - alpha) * 0.25; ...   % left neighbour
            left_idx,      (1 - alpha) * 0.50; ...   % left of centre
            left_idx + 1,   alpha      * 0.50; ...   % right of centre
            left_idx + 2,   alpha      * 0.25  ...   % right neighbour
        ];

        for k = 1:size(idx_weights, 1)
            j = idx_weights(k, 1);   % fine-grid column index
            w = idx_weights(k, 2);   % corresponding weight
            if j >= 1 && j <= N_fine % skip out-of-bounds (boundary handling)
                rows(end+1) = i;     %#ok<AGROW>
                cols(end+1) = j;     %#ok<AGROW>
                vals(end+1) = w;     %#ok<AGROW>
            end
        end
    end

    R = sparse(rows, cols, vals, N_coarse, N_fine);

    % Row-normalise: ensures a constant fine-grid signal -> same constant on coarse.
    % Boundary rows may have fewer than 4 contributing fine points, so their
    % raw weights do not sum to 1. Normalisation corrects this.
    row_sums = sum(R, 2);
    D = spdiags(1 ./ max(row_sums, eps), 0, N_coarse, N_coarse);
    R = D * R;
end
