function [xs, obj] = func_admm_reg(x, G, R, yi, ri, rho, xiter, niter, alpha)


%% initialization of variables 
ld = zeros(size(yi));
Gx = G*x;
v = Gx;

xs(:,:,:,1) = x;
C = R.C; % finite differencing matrix 

%% indexing nonzero projection 
ii = yi == 0;
iii = yi > 0;

for k = 1:niter
    
    %% x-update
    for l = 1:xiter        
        grad = G' * (G * x - v + ld/rho); 
        grad = rho * grad + R.cgrad(R, x); % (48) in paper 
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
            step = ((grad2') * grad2) / denom; % (49) in paper 
            step = real(step); 
        end        
        x = x - grad * step;
    end
    xs = cat(4,xs,x);
    figure(7); im(x); drawnow;
    
    %% v-update 
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
    vv = v + ri * alpha;
    vv(vv<0) = 0;
    v = vv - ri * alpha;
    
    min_yi1 = v(iii) + ri(iii);
    min_yi0 = v(ii) + ri(ii);
    
    %% calculating cost function 
    yp = Gx + ri;
    good = yi > 0;
    vox = sum(yp(good) <= 0);
    if vox > 0
        yiyp = yi.*yp;
        good = yiyp > 0;
    end
    
    like = sum(yp(:)) - sum(yi(good) .* log(yp(good))); % new better way
    penal = R.penal(R, x);
    obj(k) = like + penal;
    
    %% adaptive choice of admm penalty parameter 
    delta_g_hat = -(v - v_old);
    primal_residual = Gx - v;
    primal_residual_list(k) = norm(primal_residual(:));
    dual_residual = rho * G' * delta_g_hat(:);
    dual_residual_list(k) = norm(dual_residual(:));     
    if primal_residual_list(k) > 10*dual_residual_list(k)
        rho = rho * 2;
    elseif dual_residual_list(k) > 10*primal_residual_list(k)
        rho = rho / 2;
    else
        rho = rho;
    end
    
    %% lambda-update
    ld = double(ld + rho*(Gx - v));
    
    printf('Iteration: %g, Range: %g %g, Likelihood:%g, Obj: %g, rho:%g, P_Res:%g, D_Res:%g', ...
        k, min(x(:)), max(x(:)), like, obj(k), rho, primal_residual_list(k), dual_residual_list(k))
    
end


