function [x_fore,F] = BVAR_FORE(x,T,r,k,g,ng,p,nfore,interF,var_sv,nsave,nburn,nthin)

%% BVAR_FORE.m   Bayesian VAR Model (forecasting function)
% =========================================================================
% Written by: 
%
%      Dimitris Korobilis         and       Maximilian Schroeder
%    University of Glasgow        and   Norwegian BI Business School
%
% First version: 06 July 2022
% This version: 05 November 2023
% =========================================================================

Flag = mlag2([x(:,1:r),g],p);                               % Lags of factors for VAR part (DFM state equation)
Fy = [x(p+1:end,1:r), g(p+1:end,:)];          % LHS variables for state equation (correct observations for number of lags)
Fx = [ones(T-p,0+interF) Flag(p+1:end,:)];       % RHS variables for state equation (correct observations for number of lags)
resid = zeros(T-p,r+ng);
A_ = eye(r+ng);

%=========| Estimation
% Horseshoe prior for Phi
lambdaPhi = 0.1*ones(r+ng,k);                 % "local" shrinkage parameters
tauPhi    = 0.1*ones(r+ng,1);                 % "global" shrinkage parameter
nuPhi     = 0.1*ones(r+ng,k);  
xiPhi     = 0.1*ones(r+ng,1); 

% Choose sampling algorithm for VAR parameters
est_meth = 1 + double(k>T);

% Initialize matrices
Phi       = 0.1*ones(k,(r+ng));
Omega     = 0.1*ones(1,r+ng);
Omega_t   = 0.1*ones(T-p,r+ng);
OMEGA     = 0.1*ones(r+ng,r+ng,T);
h         = 0.1*ones(T-p,r+ng);   
sig       = 0.1*ones(r+ng,1);
Omegac    = zeros((r+ng)*p,(r+ng)*p,T);
Phic      = [Phi(interF+1:end,:)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1),r+ng)]; 
Omegac(1:r+ng,1:r+ng,:) = repmat(diag(Omega),1,1,T); 
QPhi      = 1*ones(k,(r+ng));
intF      = zeros(T,(r+ng)*p);

% Storage space for Gibbs draws
Phi_draws   = zeros(k,r+ng,nsave/nthin);
OMEGA_draws = zeros(r+ng,r+ng,T,nsave/nthin);

xfore_save  = zeros(r+ng,nfore,nsave/nthin);
%% ============================| START MCMC |==============================
format bank;
fprintf('Now you are running BVAR with MCMC')
fprintf('\n')
fprintf('Iteration 000000')
savedraw = 0;
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,10) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%6d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end
    
    %% =====| Step 5: Sample VAR variances Omega
    se = (Fy - Fx*Phi).^2;
    if var_sv == 0
        b1 = 0.01 + (T-p)/2; 
        b2 = 0.01 + sum(se)/2;             
        Omega(1,:) = 1./gamrnd(b1,1./b2);                % Sample Omega from inverse-Gamma
        Omega_t = repmat(Omega,T-p,1);
    elseif var_sv == 1
        fystar  = log(se + 1e-6);                        % log squared residuals
        for i = 1:r+ng
            [h(:,i), ~] = SVRW(fystar(:,i),h(:,i),sig(i,:),4);                   % log stochastic volatility using Chan's filter   
            Omega_t(:,i)  = exp(h(:,i));                               % convert log-volatilities to variances
            r1 = 1 + (T-p-1)/2;   r2 = 0.01 + sum(diff(h(:,i)).^2)'/2;  % posterior moments of variance of log-volatilities
            sig(i,:) = 1./gamrnd(r1./2,2./r2);
        end
    end

    %% =====| Step 6: Sample VAR coefficients Phi
    for i = 1:r+ng                                  
        Fy_tilde = Fy(:,i)./sqrt(Omega_t(:,i));                 % Standardized LHS variables
        FX_tilde = [Fx resid(:,1:i-1)]./sqrt(Omega_t(:,i));     % Standardized RHS variables
        VAR_coeffs = randn_gibbs(Fy_tilde,FX_tilde,[QPhi(:,i);9*ones(i-1,1)],T-p,k+i-1,est_meth);   % Sample VAR coefficients Phi
        Phi(:,i) = VAR_coeffs(1:k);  
        A_(i,1:i-1) = VAR_coeffs(k+1:end);        
        [QPhi(:,i),~,~,lambdaPhi(i,:),tauPhi(i,1),nuPhi(i,:),xiPhi(i,1)] = horseshoe_prior(Phi(:,i)',k,tauPhi(i,1),nuPhi(i,:),xiPhi(i,1));  % sample prior variance of loadings Lambda
        resid(:,i) = Fy(:,i) - [Fx resid(:,1:i-1)]*VAR_coeffs;
    end   
    Phic = [Phi(interF+1:end,:)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1),r+ng)];       % VAR coefficients in companion form
    % Ensure stationary draws
    while max(abs(eig(Phic)))>0.999
        for i = 1:r+ng
            Fy_tilde = Fy(:,i)./sqrt(Omega_t(:,i));             % Standardized LHS variables
            FX_tilde = [Fx resid(:,1:i-1)]./sqrt(Omega_t(:,i)); % Standardized RHS variables
            VAR_coeffs = randn_gibbs(Fy_tilde,FX_tilde,[QPhi(:,i);9*ones(i-1,1)],T-p,k+i-1,1);   % Sample VAR coefficients Phi  
            Phi(:,i) = VAR_coeffs(1:k);  
            A_(i,1:i-1) = VAR_coeffs(k+1:end);
            [QPhi(:,i),~,~,lambdaPhi(i,:),tauPhi(i,1),nuPhi(i,:),xiPhi(i,1)] = horseshoe_prior(Phi(:,i)',k,tauPhi(i,1),nuPhi(i,:),xiPhi(i,1));  % sample prior variance of loadings Lambda
            resid(:,i) = Fy(:,i) - [Fx resid(:,1:i-1)]*VAR_coeffs;
        end
    Phic = [Phi(interF+1:end,:)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1),r+ng)];       % VAR coefficients in companion form
    end
    intF(:,1:r+ng) = (interF==1)*repmat(Phi(1,:),T,1);
    OMEGA(:,:,1:p) = repmat(A_*diag(Omega_t(1,:))*A_',1,1,p);
    for t = 1:T-p    
        OMEGA(:,:,t+p) = A_*diag(Omega_t(t,:))*A_';
    end
    Omegac(1:r+ng,1:r+ng,:) = OMEGA;         
    
    %% Do stuff after burn-in period has passed and nburn samples are discarted
    if irep > nburn && mod(irep,nthin)==0
        % Save draws of parameters
        savedraw = savedraw + 1;
        Phi_draws(:,:,savedraw)     = Phi;
        OMEGA_draws(:,:,:,savedraw) = OMEGA;
                      
        %% ======================================================================================================
        %% =================================| Quantile level forecasting |=======================================
        % first obtain nfore VAR forecasts from the state equation
        fx_fore = [Fy(end,:) Flag(end,1:((r+ng)*(p-1)))];
        miu = intF(1,:)';
        fy_fore = zeros(nfore,r+ng);
        VAR_MEAN = 0;
        for ifore = 1:nfore              
            VAR_MEAN =  VAR_MEAN + (Phic^(ifore-1))*miu;   
            FORECASTS = VAR_MEAN + (Phic^ifore)*fx_fore';          
            fy_fore(ifore,:) = FORECASTS(1:r+ng,:)'; 
        end
        xfore_save(:,:,savedraw) = fy_fore';
    end
end

x_fore = xfore_save;