function L_coarse = restrict_matrix(L_fine, Ntheta, Ntau, Ntheta_c, Ntau_c, Nx, Ny, Nx_c, Ny_c)
    % Restrict fine system matrix L_fine of size (Ntheta*Ntau) x (Nx*Ny)
    % to coarse system matrix L_coarse of size (Ntheta_c*Ntau_c) x (Nx_c*Ny_c)
    %
    % Inputs:
    %   L_fine  - fine system matrix
    %   Ntheta, Ntau        - fine sinogram dimensions
    %   Ntheta_c, Ntau_c    - coarse sinogram dimensions
    %   Nx, Ny              - fine image dimensions
    %   Nx_c, Ny_c          - coarse image dimensions
    %
    % Output:
    %   L_coarse - coarse system matrix

    % Build restriction operator for sinogram (rows)
    Rs = build_fw_2D(Ntheta, Ntau, Ntheta_c, Ntau_c);
    % Build prolongation operator for image (columns)
    Px = build_fw_2D(Nx, Ny, Nx_c, Ny_c)'; % transpose of restriction

    % Restrict operator: L_c = Rs * L_fine * Px
    L_coarse = Rs * L_fine * Px;
end


