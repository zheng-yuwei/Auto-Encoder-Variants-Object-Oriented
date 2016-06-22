function KLDeriv = getKL_deriv(sparse_rho,rho_hat)
%KL-散度函数的导数
    EPSILON = 1e-8; %防止除0
    KLDeriv = ( -sparse_rho ) ./ ( rho_hat + EPSILON ) + ...
        ( 1 - sparse_rho ) ./ ( 1 - rho_hat + EPSILON );  
end