function x_coarse = restrict_vector(x, Nx, Ny, Nx_c, Ny_c)
    % Restrict vectorized image x (size Nx*Ny x 1) to coarse size Nx_c x Ny_c
    % using full weighting operator adapted for arbitrary sizes.
    %
    % Inputs:
    %   x    - vectorized fine image (Nx*Ny x 1)
    %   Nx    - fine image rows
    %   Ny    - fine image cols
    %   Nx_c  - coarse image rows
    %   Ny_c  - coarse image cols
    %
    % Output:
    %   x_coarse - vectorized coarse image (Nx_c*Ny_c x 1)

    assert(Nx_c <= Nx && Ny_c <= Ny, 'Coarse size must be smaller or equal');

    % Build 1D full weighting restriction operators for each dimension
    Rx = build_fw_1D(Nx, Nx_c);
    Ry = build_fw_1D(Ny, Ny_c);

    % 2D full weighting operator by Kronecker product
    Rx_flat = kron(Ry, Rx);

    % Apply restriction
    x_coarse = Rx_flat * x;
end




