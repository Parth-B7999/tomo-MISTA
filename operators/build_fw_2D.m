function R = build_fw_2D(Nx_f, Ny_f, Nx_c, Ny_c)
    % Builds 2D full weighting restriction operator for arbitrary sizes
    % using Kronecker product of 1D full weighting operators

    Rx = build_fw_1D(Nx_f, Nx_c);
    Ry = build_fw_1D(Ny_f, Ny_c);
    R = kron(Ry, Rx);
end