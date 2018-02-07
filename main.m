clc
clear

%% read XCAT data 
mumap = bin_read('data/Y90PET_atn_1.bin'); % reading attenuation map 
mumap = mumap / 10; % cm -> mm
ind = bin_read('data/Y90PET_act_1.bin');
liver = single((ind == 13)); % setting liver mask 
llung = (ind == 15); rlung = (ind == 16); % setting lung mask 
lung = single(llung + rlung);

%% image/projection geometry setting
nx = 128; % number of voxel in x direction 
ny = 128; % number of voxel in y direction 
nz = 100; % number of voxel in z direction 
dx = 4; % voxel size in x direction
dz = 4; % voxel size in z direction
na = 168; % number of projection angle
    
%% recon parameter setting
rho = 1; % initial rho value 
patient_A = 1; % 1 if Patient A condition, 0 if Patient B condition
beta = 2^-3; % regularization parameter 
alpha = 1; % constraint: Ax + alpha * r > 0 
psi = 1; % negml poisson-gaussian switching parameter 

xiter = 1; % number of x inner update 
niter = 50; % number of admm update 
titer = xiter * niter; % total number of iterations 

%% setting 
ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', dx, 'dz', dz); 
ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-4)) > 0;
sg = sino_geom('par', 'nb', ig.nx, 'na', na * ig.nx / nx, ...
    'dr', ig.dx, 'strip_width', 2*ig.dx);
R = Reg1(ig.mask, 'edge_type', 'tight', ...
    'beta', beta, 'pot_arg', {'quad'},'type_denom', 'matlab');

%% Patient A or B condition 
if patient_A
    f.counts = 6e5;
    f.scatter_percent = 500;
else % We use smaller area of liver in the Patient B case
    f.counts = 1e5;
    f.scatter_percent = 1800;
    liver(64:128,:,:) = 0;
    liver(:,76:128,:) = 0;
    liver(:,1:50,:) = 0;
    liver(:,:,1:55) = 0;
    liver(:,:,73:end) = 0;
end


%% setting hot spot/cold spot/healthy liver/eroded liver
hotspot = ellipsoid_im(ig, ...
    [-40 30 50  20 20 25    0 0 1;
    ], 'oversample', 1);
coldspot = ellipsoid_im(ig, ...
    [-90 -10 60  20 20 25    0 0 1;
    ], 'oversample', 1);
hliver = liver - hotspot - coldspot;
hliver(hliver < 0) = 0;
se = strel('arbitrary',eye(2));
eliver = imerode(hliver,se);

%% true image 
xtrue = double(liver + 4*hotspot - coldspot + 0.04*lung);
xtrue(xtrue < 0) = 0;

%% system model
f.dir = test_dir;
f.dsc = [test_dir 't.dsc'];
f.wtr = strrep(f.dsc, 'dsc', 'wtr');
f.mask = [test_dir 'mask.fld'];
fld_write(f.mask, ig.mask)

tmp = Gtomo2_wtmex(sg, ig, 'mask', ig.mask_or);
[tmp dum dum dum dum is_transpose] = ...
    wtfmex('asp:mat', tmp.arg.buff, int32(0));
if is_transpose
    tmp = tmp'; % because row grouped
end
delete(f.wtr)
wtf_write(f.wtr, tmp, ig.nx, ig.ny, sg.nb, sg.na, 'row_grouped', 1)

f.sys_type = sprintf('2z@%s@-', f.wtr);

G = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, ...
    'chat', 0, 'view2d', 1, 'nthread', jf('ncore'));

%% simulate noisy projection data
ytrue = G * xtrue;
li = G * mumap;
ci = exp(-li);
ci = ci * f.counts / sum(col(ci .* ytrue));
ytrue = ci .* ytrue;
ri = ones(size(ytrue)) * f.scatter_percent / 100 * mean(ytrue(:));
Gb = Gblock(G, 1, 'odiag', ci);
ci = ones(size(ri));
yi = poisson(ytrue + ri);
xinit = double6(ig.mask); xinit = xinit/sum(xinit(:)); % initial uniform image 

%% recon using our proposed method / regularized negml / regularized sps 
[xadmm, obj_admm] = func_admm_reg(xinit, Gb, R, yi, ri, rho, xiter, niter, alpha);
xneg_reg = eql_neg_reg(xinit(ig.mask), Gb, R, psi, yi(:), ri(:), ig, titer);
xsps_reg = eql_sps_os(xinit(ig.mask), Gb, reshaper(yi, '2d'), reshaper(ci, '2d'), reshaper(ri, '2d'), R, titer, inf, 'oc');
xsps_reg = ig.embed(xsps_reg);