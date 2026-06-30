function [x_fore] = QFAVAR_FORE2(x,xlag,T,n,k,g,ng,r,p,quant,nfore,interX,interF,AR1x,incldg,var_sv,ALsampler,nsave,nburn,nthin)

%% QFAVAR_FORE.m  Bayesian Quantile Factor-Augmented VAR Model (forecasting function)
% =========================================================================
% Written by: 
%
%      Dimitris Korobilis         and       Maximilian Schroeder
%    University of Glasgow        and   Norwegian BI Business School
%
% First version: 06 July 2022
% This version: 05 November 2023
% =========================================================================

%=========| Estimation
nq        = length(quant);  % Number of quantiles
% The next three quantities are required for the posterior of latent quantities z and w
k2_sq     = 2./(quant.*(1-quant));
k1        = (1-2*quant)./(quant.*(1-quant));
k1_sq     = k1.^2;

% Horseshoe prior for L
lambdaL   = 0.1*ones(n,r+AR1x+interX+ng,nq);     % "local" shrinkage parameters
tauL      = 0.1*ones(n,nq);                      % "global" shrinkage parameter
nuL       = 0.1*ones(n,r+AR1x+interX+ng,nq);  
xiL       = 0.1*ones(n,nq); 

% Horseshoe prior for Phi
lambdaPhi = 0.1*ones(r*nq+ng,k);                 % "local" shrinkage parameters
tauPhi    = 0.1*ones(r*nq+ng,1);                 % "global" shrinkage parameter
nuPhi     = 0.1*ones(r*nq+ng,k);  
xiPhi     = 0.1*ones(r*nq+ng,1); 

% Choose sampling algorithm for VAR parameters
est_meth = 1 + double(k>T);

% Initialize matrices
xbar      = 0*x;
Lbar      = zeros(n,r,T,nq);
Lbar2     = zeros(n*nq,ng,T);
L         = zeros(r+AR1x+interX+ng,n,nq);
Sigma     = 0.1*ones(n,nq);
z         = 0.1*ones(T,n,nq);
Phi       = 0.1*ones(k,(r*nq+ng));
Omega     = 0.1*ones(1,r*nq+ng);
Omega_t   = 0.1*ones(T-p,r*nq+ng);
OMEGA     = 0.1*ones(r*nq+ng,r*nq+ng,T);
h         = 0.1*ones(T-p,r*nq+ng);   
sig       = 0.1*ones(r*nq+ng,1);
F         = zeros(T,r*nq+ng);
FL        = zeros(T,n,nq);  
Omegac    = zeros((r*nq+ng)*p,(r*nq+ng)*p,T);
Phic      = [Phi(interF+1:end,:)'; eye((r*nq+ng)*(p-1)) zeros((r*nq+ng)*(p-1),r*nq+ng)]; 
Omegac(1:r*nq+ng,1:r*nq+ng,:) = repmat(diag(Omega),1,1,T); 
QL        = 1*ones(r+AR1x+interX+ng,n,nq);
QPhi      = 1*ones(k,(r*nq+ng));
intF      = zeros(T,(r*nq+ng)*p);

% Extract FA and QFA estimates (using PCA and VBQFA)
fpca      = zeros(T,r);  
fqfa      = zeros(T,r,nq);
%disp('Extracting PCA and VBQFA factors...')
for ifac = 1:r
   fpca(:,ifac) = extract(zscore(x(:,(ifac-1)*9+1:ifac*9)),1);
   fqfa(:,ifac,:) = VBQFA(zscore(x(:,(ifac-1)*9+1:ifac*9)),1,500,quant,0,1);
end
F(:,1:r*nq) = fqfa(:,:);
F = zscore(F);

if AR1x == 0; ARindex = []; else, ARindex=1; end

% Storage space for Gibbs draws
F_draws     = zeros(T,r*nq,nsave/nthin);
L_draws     = zeros(r+AR1x+interX+ng,n,nq,nsave/nthin);
Phi_draws   = zeros(k,r*nq+ng,nsave/nthin);
Sigma_draws = zeros(n,nq,nsave/nthin);
OMEGA_draws = zeros(r*nq+ng,r*nq+ng,T,nsave/nthin);
z_draws     = zeros(T,n,nq,nsave/nthin); 

xfore_save  = zeros(n,nq,nfore,nsave/nthin);
%% ============================| START MCMC |==============================
format bank;
fprintf('\n')
fprintf('Now you are running QFAVAR with MCMC')
fprintf('\n')
fprintf('Iteration 000000')
savedraw = 0;
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,10) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%6d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end

    %%%%%%%% ================================================================================================== %%%%%%%%
    %%%%%%%% =====================| QFAVAR measurement equation (Factor extraction) |========================== %%%%%%%%
    Lc    = zeros(n*nq,r*nq+ng,T);
    Lfull = zeros(n*nq+ng,r*nq+ng,T);
    for q = 1:nq
        x_tilde = 0*x;
        Fq = F(:,(q-1)*r+1:q*r);
        for i = 1:n  % equation-by-equation estimation
            %% Select the factors that correspond to the i-th variable
            F_all = [ones(T,(interX==1)), xlag(:,ARindex*i), g, Fq];
            select = [1:interX+AR1x + incldg*ng, interX+AR1x+ng+floor((i-1)/9+1)];
            F_select = F_all(:,select);

            %% =====| Step 1: Sample loadings L
            v = sqrt(Sigma(i,q)*k2_sq(q)*z(:,i,q));                                                    % This is the variance of the Asymetric Laplace errors
            x_tilde(:,i) = (x(:,i) - k1(q).*z(:,i,q))./v;                                              % Standardized LHS variables
            F_tilde = F_select./v;                                                                     % Standardized RHS variables                     
            L(select,i,q) = randn_gibbs(x_tilde(:,i),F_tilde,QL(select,i,q),T,length(select),1);       % Sample loadings Lambda
            [QL(select,i,q),~,~,lambdaL(i,select,q),tauL(i,q),nuL(i,select,q),xiL(i,q)] = ...
                horseshoe_prior(L(select,i,q)',length(select),tauL(i,q),nuL(i,select,q),xiL(i,q));   % sample prior variance of loadings Lambda

            %% =====| Step 2: Sample latent indicators z    
            FL(:,i,q) = F_all*L(:,i,q);
            if ALsampler == 1             % Khare and Robert (2012)                          
                chi_z = sqrt(k1_sq(q) + 2*k2_sq(q))./abs(x(:,i) - squeeze(FL(:,i,q)));      % posterior moment k1 of z
                psi_z = (k1_sq(q) + 2*k2_sq(q))./(Sigma(i,q).*k2_sq(q));                    % posterior moment k2 of z
                z(:,i,q)  = min(1./random('InverseGaussian',chi_z,psi_z,T,1),1e+6);         % Sample z from Inverse Gaussian
            elseif ALsampler == 2         % Kozumi and Kobayashi (2011)
                chi_z = ((x(:,i) - squeeze(FL(:,i,q))).^2)./(Sigma(i,q).*k2_sq(q));         % posterior moment k1 of z
                psi_z = (k1_sq(q) + 2*k2_sq(q))./(Sigma(i,q).*k2_sq(q));                    % posterior moment k2 of z
                for t = 1:T
                    z(t,i,q)  = min(gigrnd(0.5,psi_z,chi_z(t),1),1e+6);                     % Sample z from Generalized Inverse Gaussian
                end
            end

            %% =====| Step 3: Sample factor regression variances Sigma
            a1 = 0.01 + 3*T/2;      sse = (x(:,i) - FL(:,i,q) - k1(q).*z(:,i,q)).^2;
            a2 = 0.01 + sum(sse./(2*z(:,i,q).*k2_sq(q))) + sum(z(:,i,q));       
            Sigma(i,q) = 1./gamrnd(a1,1./a2);                                         % Sample Sigma from inverse-Gamma
        end

        % Normalize loadings
%         for ir = 1:r
%             L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9,q) = L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9,q)./L(interX+AR1x+ng+ir,(ir-1)*9+q,q);
%         end

        % Create matrices for augmented state-space form (needed for sampling factors F)
        for i = 1:n
            Ftemp = [ones(T,(interX==1)), xlag(:,ARindex*i)];
            xbar(:,i,q) = (x(:,i) - Ftemp*L(1:interX+AR1x,i,q) - k1(q).*z(:,i,q))./sqrt(k2_sq(q)*z(:,i,q));
            Lbar(i,1:r,:,q) = L(interX+AR1x+ng+1:end,i,q)./sqrt(k2_sq(q)*z(:,i,q))';
            Lbar2((q-1)*n+i,:,:) = L(interX+AR1x+1:interX+AR1x+ng,i,q)./sqrt(k2_sq(q)*z(:,i,q))';
        end
        % Lc has a diagonal block with all L matrices for all quantiles
        Lc((q-1)*n+1:q*n,(q-1)*r+1:q*r,:) = Lbar(:,:,:,q);
    end
    % In the QFAVAR, Lc also has the last ng columns corresponding to variables g_{t}
    for t = 1:T
        Lc(:,end-ng+1:end,t) = Lbar2(:,:,t);
        Lfull(:,:,t) = [Lc(:,:,t); zeros(ng,r*nq) eye(ng)];
    end

%     %% =====| Step 4: Sample factors [F;g]
%     [F] = FFBS([xbar(:,:) g],Lfull,intF,Phic,[Sigma(:);zeros(ng,1)],Omegac,(r*nq+ng));    % Forward sampling backward smoothing algorithm
%     % % Standardize factors
%     % F = zscore(F);
%     
%     % Make sure factors are sign rotated, so that the same factor in different quantiles has the same interpretation
%     % (factors are not sign identified between different quantile levels)
%     for q = 1:nq
%         for ir = 1:r
%             Ctemp =  corrcoef([F(:,(q-1)*r+ir) fqfa(:,ir,q)]);
%             F(:,(q-1)*r+ir) =  F(:,(q-1)*r+ir).*sign(Ctemp(1,2));
%             L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9,q) = L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9,q).*sign(Ctemp(1,2));
%         end
%     end

    %%%%%%%% ================================================================================================== %%%%%%%%
    %%%%%%%% ================================| State equation (VAR dynamics) |================================= %%%%%%%%
    Flag = mlag2([F(:,1:nq*r),g],p);                               % Lags of factors for VAR part (DFM state equation)
    Fy = [F(p+1:end,1:r*nq), g(p+1:end,:)];          % LHS variables for state equation (correct observations for number of lags)
    Fx = [ones(T-p,0+interF) Flag(p+1:end,:)];       % RHS variables for state equation (correct observations for number of lags)
    resid = zeros(T-p,r*nq+ng);
    A_ = eye(r*nq+ng);
    
    %% =====| Step 5: Sample VAR variances Omega
    se = (Fy - Fx*Phi).^2;
    if var_sv == 0
        b1 = 0.01 + (T-p)/2; 
        b2 = 0.01 + sum(se)/2;             
        Omega(1,:) = 1./gamrnd(b1,1./b2);                % Sample Omega from inverse-Gamma
        Omega_t = repmat(Omega,T-p,1);
    elseif var_sv == 1
        fystar  = log(se + 1e-6);                        % log squared residuals
        for i = 1:r*nq+ng
            [h(:,i), ~] = SVRW(fystar(:,i),h(:,i),sig(i,:),4);                   % log stochastic volatility using Chan's filter   
            Omega_t(:,i)  = exp(h(:,i));                               % convert log-volatilities to variances
            r1 = 1 + (T-p-1)/2;   r2 = 0.01 + sum(diff(h(:,i)).^2)'/2;  % posterior moments of variance of log-volatilities
            sig(i,:) = 1./gamrnd(r1./2,2./r2);
        end
    end

    %% =====| Step 6: Sample VAR coefficients Phi
    for i = 1:r*nq+ng                                  
        Fy_tilde = Fy(:,i)./sqrt(Omega_t(:,i));                 % Standardized LHS variables
        FX_tilde = [Fx resid(:,1:i-1)]./sqrt(Omega_t(:,i));     % Standardized RHS variables
        VAR_coeffs = randn_gibbs(Fy_tilde,FX_tilde,[QPhi(:,i);9*ones(i-1,1)],T-p,k+i-1,est_meth);   % Sample VAR coefficients Phi
        Phi(:,i) = VAR_coeffs(1:k);  
        A_(i,1:i-1) = VAR_coeffs(k+1:end);        
        [QPhi(:,i),~,~,lambdaPhi(i,:),tauPhi(i,1),nuPhi(i,:),xiPhi(i,1)] = horseshoe_prior(Phi(:,i)',k,tauPhi(i,1),nuPhi(i,:),xiPhi(i,1));  % sample prior variance of loadings Lambda
        resid(:,i) = Fy(:,i) - [Fx resid(:,1:i-1)]*VAR_coeffs;
    end   
    Phic = [Phi(interF+1:end,:)'; eye((r*nq+ng)*(p-1)) zeros((r*nq+ng)*(p-1),r*nq+ng)];       % VAR coefficients in companion form
    % Ensure stationary draws
    while max(abs(eig(Phic)))>0.999
        for i = 1:r*nq+ng
            Fy_tilde = Fy(:,i)./sqrt(Omega_t(:,i));             % Standardized LHS variables
            FX_tilde = [Fx resid(:,1:i-1)]./sqrt(Omega_t(:,i)); % Standardized RHS variables
            VAR_coeffs = randn_gibbs(Fy_tilde,FX_tilde,[QPhi(:,i);9*ones(i-1,1)],T-p,k+i-1,1);   % Sample VAR coefficients Phi  
            Phi(:,i) = VAR_coeffs(1:k);  
            A_(i,1:i-1) = VAR_coeffs(k+1:end);
            [QPhi(:,i),~,~,lambdaPhi(i,:),tauPhi(i,1),nuPhi(i,:),xiPhi(i,1)] = horseshoe_prior(Phi(:,i)',k,tauPhi(i,1),nuPhi(i,:),xiPhi(i,1));  % sample prior variance of loadings Lambda
            resid(:,i) = Fy(:,i) - [Fx resid(:,1:i-1)]*VAR_coeffs;
        end
    Phic = [Phi(interF+1:end,:)'; eye((r*nq+ng)*(p-1)) zeros((r*nq+ng)*(p-1),r*nq+ng)];       % VAR coefficients in companion form
    end
    intF(:,1:r*nq+ng) = (interF==1)*repmat(Phi(1,:),T,1);
    OMEGA(:,:,1:p) = repmat(A_*diag(Omega_t(1,:))*A_',1,1,p);
    for t = 1:T-p    
        OMEGA(:,:,t+p) = A_*diag(Omega_t(t,:))*A_';
    end
    Omegac(1:r*nq+ng,1:r*nq+ng,:) = OMEGA;         
    
    %% Do stuff after burn-in period has passed and nburn samples are discarted
    if irep > nburn && mod(irep,nthin)==0
        % Save draws of parameters
        savedraw = savedraw + 1;
        F_draws(:,:,savedraw)       = F(:,1:nq*r);
        L_draws(:,:,:,savedraw)     = L;
        Phi_draws(:,:,savedraw)     = Phi;
        Sigma_draws(:,:,savedraw)   = Sigma;
        OMEGA_draws(:,:,:,savedraw) = OMEGA;
        z_draws(:,:,:,savedraw)     = z;        

        %% ======================================================================================================
        %% =================================| Quantile level forecasting |=======================================
        % first obtain nfore VAR forecasts from the state equation
        fx_fore = [Fy(end,:) Flag(end,1:((r*nq+ng)*(p-1)))];
        miu = intF(1,:)';
        fy_fore = zeros(nfore,r*nq+ng);
        VAR_MEAN = 0;
        for ifore = 1:nfore                 
            VAR_MEAN =  VAR_MEAN + (Phic^(ifore-1))*miu;   
            FORECASTS = VAR_MEAN + (Phic^ifore)*fx_fore';          
            fy_fore(ifore,:) = FORECASTS(1:r*nq+ng,:)'; 
        end
        % next project quantile forecasts in the measurement equation
        xfore = zeros(n,nq,nfore);
        for q = 1:nq   
            for ifore = 1:nfore
                Fq = fy_fore(ifore,(q-1)*r+1:q*r);
                for i = 1:n
                    if ifore == 1
                        f_fore = [interX*1, AR1x*x(end,i), fy_fore(ifore,r*nq+1:end), Fq(:,floor((i-1)/9+1))];
                    else
                        f_fore = [interX*1, AR1x*xfore(i,q,ifore-1), fy_fore(ifore,r*nq+1:end), Fq(:,floor((i-1)/9+1))]; 
                    end
                    f_fore(:,sum(f_fore,1)==0)=[];
                    select = [1:interX+AR1x+ng, interX+AR1x+ng+floor((i-1)/9+1)];
                    xfore(i,q,ifore) =  f_fore*L(select,i,q);
                end   
            end
        end
        xfore_save(:,:,:,savedraw) = xfore;
    end
end

x_fore = squeeze(mean(xfore_save,4));