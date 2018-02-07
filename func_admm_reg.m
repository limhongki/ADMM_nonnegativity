function [xs, obj] = func_admm_reg(x, G, W, R, yi, ri, ig, rho, xiter, niter, eps_cor, alpha)

[m,n,p] = size(yi);
ld = double(zeros(m,n,p));

ii = yi == 0;
iii = yi > 0;

Gx = G*x;
v = Gx; 

% ui = ld/rho;
% v_yi0 = Gx + ui - 1/rho;
% v(ii) = v_yi0(ii);
% 
% bb = -ui - Gx +ri + 1/rho;
% bb = 1/2*bb;
% cc = ri.*(-ui - Gx) + (ri-yi)/rho;
% cc = -cc;
% v_yi1 = real(eql_root(ones(size(bb)),bb,cc));
% v(iii) = v_yi1(iii);
% vv = v + ri * alpha;
% vv(vv<0) = 0;
% v = vv - ri * alpha;
    


xs(:,:,:,1) = x;
C = R.C;


for k = 1:niter
%     x = eql_sps_admm_reg(x(ig.mask), G, R, v(:), ld(:)/rho, rho, ig, xiter);
%     x = qpwls_psd1(x(ig.mask), G, W, v(:)-ld(:)/rho, R.C, rho, R, 'niter', xiter);
     grad = G' * (G * x - v + ld/rho);
     grad = rho * grad + R.cgrad(R, x);
     grad2 = grad(:);
     Adir = G * grad;
     Adir = Adir(:);
     Cdir = C * grad;
     Cdir = Cdir(:);
     denom = rho*(Adir')*Adir + Cdir'*Cdir;
     denom = reale(denom, 'error', 'denom');
     if denom == 0
         warning 'found exact solution??? step=0 now!?'
         step = 0;
     else
         denom = reale(denom, 'error', 'denom');
         step = ((grad2') * grad2) / denom;
         step = real(step); % see a-complex.pdf
     end
     
    x = x - grad * step;
%     x = ig.embed(x);
    xs = cat(4,xs,x); 
    x = x(:,:,:,end);
    figure(10); im(x); drawnow;
    Gx = G*x;
    
    ui = ld/rho;
    
    v_old = v;
    
    v_yi0 = Gx + ui - 1/rho;
    bb = -ui - Gx + ri + 1/rho;
    bb = 1/2*bb;
    cc = ri.*(-ui - Gx) + (ri-yi)/rho;
    cc = -cc;
    v_yi1 = real(eql_root(ones(size(bb)),bb,cc));
    
    v(ii) = v_yi0(ii);
    v(iii) = v_yi1(iii);
%     v = real(eql_root(ones(size(bb)),bb,cc));
    vv = v + ri * alpha;
    vv(vv<0) = 0;
    v = vv - ri * alpha;
    
    min_yi1 = v(iii) + ri(iii);
    min_yi0 = v(ii) + ri(ii);
    if sum(v(iii) + ri(iii) <= 0) > 0  
        break;
    end
    
    
    
    yp = Gx + ri;
    good = yi > 0;
    vox = sum(yp(good) <= 0);
    if vox > 0 
        printf('minimum of v + ri: %g(yi>0) %g(yi=0) and %g negatives in yp!', min(min_yi1(:)), min(min_yi0(:)), vox)
        yiyp = yi.*yp;
        good = yiyp > 0;
    end
     
	like = sum(yp(:)) - sum(yi(good) .* log(yp(good))); % new better way
	penal = R.penal(R, x);
    obj(k) = like + penal;
    
      

    delta_g_hat = -(v - v_old);
    primal_residual = Gx - v;
    primal_residual_list(k) = norm(primal_residual(:));
    dual_residual = rho * G' * delta_g_hat(:);
    dual_residual_list(k) = norm(dual_residual(:));
    
%     if primal_residual_list(k) == 0 || dual_residual_list(k) == 0
%         rho = (primal_residual_list(k)+10^-3)/(dual_residual_list(k)+10^-3);
%     else
%         rho = (primal_residual_list(k))/(dual_residual_list(k));
%     end
%     rho = min(10^3,rho);
%     rho = max(10^-3,rho);

    if primal_residual_list(k) > 10*dual_residual_list(k)
        rho = rho * 2;
    elseif dual_residual_list(k) > 10*primal_residual_list(k)
        rho = rho / 2;
    else 
        rho = rho;
    end
    
    ld = double(ld + rho*(Gx - v));
    
    printf('Iteration: %g, Range: %g %g, Likelihood:%g, Obj: %g, rho:%g, P_Res:%g, D_Res:%g', ...
        k, min(x(:)), max(x(:)), like, obj(k), rho, primal_residual_list(k), dual_residual_list(k)) 
    
end


