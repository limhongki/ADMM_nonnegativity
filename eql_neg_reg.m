function xs = eql_neg_reg(x, Gb, R, psi, yi, ri, ig, niter)
xs = zeros(numel(x), niter+1);
xs(:,1) = x;
gi = sum(Gb')';
R2 = R.denom(R, x);
for iter = 1:niter
    li = Gb* x;
    yb = li + ri;
    num = Gb' * ((yb-yi) ./ max(psi,yb));
    R1 = R.cgrad(R, x);
    num = num + R1;
    den = Gb' * (gi ./ max(psi,yb));
    den = den + R2; 
    update = num ./ den;
    x = x - update;
    xs(:,iter+1) = x;
    if mod(iter,10) == 0
        printf('iter:%g, Range %g %g, %s', iter, min(x), max(x), mfilename)
    end
end
xs = ig.embed(xs);
