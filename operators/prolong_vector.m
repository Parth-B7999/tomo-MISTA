function x_fine = prolong_vector(x_coarse, Nx, Ny, Nx_c, Ny_c)
    % Prolong coarse vector x_coarse (size Nx_c*Ny_c x 1)
    % to fine grid vector of size Nx*Ny using transpose of full weighting operator.
    %
    % Inputs:
    %   x_coarse - vectorized coarse image (Nx_c*Ny_c x 1)
    %   Nx, Ny   - fine grid size
    %   Nx_c, Ny_c - coarse grid size
    %
    % Output:
    %   x_fine - vectorized fine image (Nx*Ny x 1)

    % Build restriction operators (same as before)
    Rx = build_fw_1D(Nx, Nx_c);
    Ry = build_fw_1D(Ny, Ny_c);

    % 2D restriction operator
    R_flat = kron(Ry, Rx);

    % Prolongation is transpose of restriction
    P_flat = R_flat';

    % Apply prolongation
    x_fine = P_flat * x_coarse;
end