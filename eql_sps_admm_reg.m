function xs = eql_sps_admm_reg(x, Gb, R, v, u, rho, ig, niter)
denom_R = R.denom(R, x);
gi = sum(Gb')';
denom_L = rho * Gb' * gi;
den = denom_L + denom_R;
for iter = 1:niter
    li = Gb* x;
    grad_L = rho * Gb' * (li - v + u);
    grad_R = R.cgrad(R, x);
    num = grad_L + grad_R;
    update = num ./ den;
    x = x - update;
    xs(:,iter) = x;
end
xs = ig.embed(xs);
% printf('Range %g %g %s', min(x), max(x), mfilename)
% printf('Took %g iterations', iter)