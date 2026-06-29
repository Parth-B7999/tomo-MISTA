function R = build_fw_1D(N_fine, N_coarse)
    % Fast build of 1D full weighting restriction matrix (N_coarse x N_fine)

    scale = N_fine / N_coarse;

    rows = [];
    cols = [];
    vals = [];

    for i = 1:N_coarse
        pos = (i - 0.5) * scale;
        left_idx = floor(pos);
        alpha = pos - left_idx;

        % Core and neighbors with weights
        idx_weights = [ ...
            left_idx - 1, (1 - alpha) * 0.25;
            left_idx,     (1 - alpha) * 0.5;
            left_idx + 1, alpha * 0.5;
            left_idx + 2, alpha * 0.25
        ];

        for k = 1:size(idx_weights,1)
            j = idx_weights(k,1);
            w = idx_weights(k,2);
            if j >= 1 && j <= N_fine
                rows(end+1) = i;
                cols(end+1) = j;
                vals(end+1) = w;
            end
        end
    end

    R = sparse(rows, cols, vals, N_coarse, N_fine);

    % Optional: normalize each row (if not guaranteed to sum to 1)
    row_sums = sum(R, 2);
    D = spdiags(1 ./ max(row_sums, eps), 0, N_coarse, N_coarse);
    R = D * R;
end


% function R = build_fw_1D(N_fine, N_coarse)
%     % Build 1D full weighting restriction matrix of size (N_coarse x N_fine)
%     % adapted for arbitrary sizes N_fine and N_coarse.
%     %
%     % This is a linear interpolation style full weighting operator.
% 
%     scale = N_fine / N_coarse;
%     R = sparse(N_coarse, N_fine);
% 
%     for i = 1:N_coarse
%         % Position of coarse point in fine grid coordinates
%         pos = (i - 0.5) * scale;
% 
%         left_idx = floor(pos);
%         alpha = pos - left_idx;
% 
%         % Weight contributions to coarse point i
%         if left_idx >= 1 && left_idx <= N_fine
%             R(i, left_idx) = R(i, left_idx) + (1 - alpha) * 0.5;
%         end
%         if left_idx + 1 >= 1 && left_idx + 1 <= N_fine
%             R(i, left_idx + 1) = R(i, left_idx + 1) + alpha * 0.5;
%         end
% 
%         % Include neighbors for full weighting
%         % Left neighbor weight 1/4 (if exists)
%         if left_idx - 1 >= 1 && left_idx - 1 <= N_fine
%             R(i, left_idx - 1) = R(i, left_idx - 1) + (1 - alpha) * 0.25;
%         end
%         % Right neighbor weight 1/4 (if exists)
%         if left_idx + 2 >= 1 && left_idx + 2 <= N_fine
%             R(i, left_idx + 2) = R(i, left_idx + 2) + alpha * 0.25;
%         end
%     end
% 
%     % Normalize rows to sum to 1 (important for stability)
%     row_sums = sum(R, 2);
%     for i = 1:N_coarse
%         if row_sums(i) > 0
%             R(i, :) = R(i, :) / row_sums(i);
%         end
%     end
% end