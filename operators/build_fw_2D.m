function R = build_fw_2D(Nx_f, Ny_f, Nx_c, Ny_c)
% build_fw_2D  Build a 2D full-weighting restriction operator.
%
% PURPOSE:
%   Restricts a 2D image (stored as a column vector of length Nx_f*Ny_f)
%   to a coarser 2D image (column vector of length Nx_c*Ny_c).
%
% CONSTRUCTION:
%   The 2D operator is the Kronecker product of two independent 1D operators:
%
%     R_2D = kron(Ry, Rx)
%
%   where Rx restricts along columns (x-direction, size Nx_c x Nx_f)
%   and   Ry restricts along rows    (y-direction, size Ny_c x Ny_f).
%
%   Kronecker structure means the operator is SEPARABLE: applying it is
%   equivalent to restricting rows first, then columns. This keeps the
%   cost O(N) instead of O(N^2).
%
%   See build_fw_1D for details of the 4-point full-weighting stencil
%   used in each direction.
%
% PROLONGATION (coarse -> fine):
%   The prolongation operator P is the TRANSPOSE of R:
%
%     P = R'    i.e.  build_fw_2D(...)'
%
%   This is the standard multigrid convention (bilinear interpolation).
%   Together: R * P ~= I on coarse space  (verified in test_02).
%
% INPUTS:
%   Nx_f, Ny_f  - fine-grid image dimensions  (columns, rows)
%   Nx_c, Ny_c  - coarse-grid image dimensions
%
% OUTPUT:
%   R - sparse restriction matrix of size  (Nx_c*Ny_c) x (Nx_f*Ny_f)
%       Assumes the image is stored column-major (MATLAB default):
%       pixel (i,j) is at index i + (j-1)*Ny_f  in the fine vector.

    % 1D restriction along x (columns) and y (rows)
    Rx = build_fw_1D(Nx_f, Nx_c);   % size: Nx_c x Nx_f
    Ry = build_fw_1D(Ny_f, Ny_c);   % size: Ny_c x Ny_f

    % Kronecker product gives the 2D separable operator
    % kron(Ry, Rx) has size (Ny_c*Nx_c) x (Ny_f*Nx_f)
    R = kron(Ry, Rx);
end
