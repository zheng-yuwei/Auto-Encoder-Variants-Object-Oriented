function KL = getKL(sparse_rho,rho_hat)
%KL-É¢¶Èº¯Êý
    EPSILON = 1e-8; %·ÀÖ¹³ý0
    KL = sparse_rho .* log( sparse_rho ./ (rho_hat + EPSILON) ) + ...
        ( 1 - sparse_rho ) .* log( (1 - sparse_rho) ./ (1 - rho_hat + EPSILON) );  
end