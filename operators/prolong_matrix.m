function L_fine = prolong_matrix(L_coarse, Ntheta, Ntau, Ntheta_c, Ntau_c, Nx, Ny, Nx_c, Ny_c)
    % Prolong coarse system matrix L_coarse to fine system matrix L_fine
    %
    % Inputs and outputs same as above but reverse
    %
    % L_fine = Rs' * L_coarse * Px'  (transpose of restrict)
    
    Rs = build_full_weighting_2D_operator(Ntheta, Ntau, Ntheta_c, Ntau_c);
    Px = build_full_weighting_2D_operator(Nx_c, Ny_c, Nx, Ny)';
    
    L_fine = Rs' * L_coarse * Px';
end
