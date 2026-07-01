%% QFAVAR_GIRF.m    Bayesian Quantile Factor-Augmented VAR Model 
%%                 (MCMC estimation + Generalized IRFs for each quantile)
%==========================================================================
% The model is of the form
%          _       _        _             _    _        _     _          _
%         |  x_{t}  |      |L(tau)  G(tau) |  |F_{t}(tau)|   | e_{t}(tau) |
%         |         |  =   |               |  |          | + |            |
%         |_ g_{t} _|      |_ 0        I  _|  |_  g_{t} _|   |_     0    _|
%         
%         _          _           _            _
%        | F_{t}(tau) |         | F_{t-1}(tau) |  
%        |            |  =  Phi |              |  + u_{t},
%        |_   g_{t}  _|         |_  g_{t-1}   _|
%
% where e_{t}(tau) ~ ALD( Sigma(tau) ) and u_{t} ~ N(0, Q(tau)), Sigma(tau) 
% and Q(tau) are diagonal covariance matrices and *tau* is the quantile level.
% =========================================================================
% Written by: 
%
%      Dimitris Korobilis         and       Maximilian Schroeder
%    University of Glasgow        and   Norwegian BI Business School
%
% First version: 06 July 2022
% This version: 05 November 2023
% =========================================================================

clear all; clc; close all;

% Add path of additional folders
addpath('functions')
addpath('data')
%---------------------------| USER INPUT |---------------------------------
% Model settings
r         = 4;              % Number of factors
p         = 2;              % Number of lags
quant     = [.1, .5, .9];   % Specify the quantiles to estimate
interX    = 0;              % Intercept in measurement equation
interF    = 0;              % Intercept in state equation
AR1x      = 0;              % Include own lag
incldg    = 1;              % Include global factors g in the measurement equation
dfm       = 0;              % 0: estimate QFAVAR; 1: estimate QDFM
standar   = 2;              % 0: no standardization; 1: standardize only x variables; 2: standardize both x and g variables
ALsampler = 1;              % Asymmetric Laplace sampler, 1: Khare and Robert (2012); 2: Kozumi and Koyabashi (2011) 
var_sv    = 0;              % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 
inflindx  = 1;              % 1: HICP Total; 2: HICP less energy; 3: HICP less food and energy
outpindx  = 2;              % 1: Unemployment; 2: Industrial Production (OECD data)
nhor      = 60;             % Horizon for IRFs, FEVDs, connectedness

% Gibbs-relaed preliminaries
nsave     = 50000;           % Number of draws to store
nburn     = 2000;           % Number of draws to discard
ngibbs    = nsave + nburn;  % Number of total draws
nthin     = 50;
iter      = 20;             % Print every "iter" iteration
%--------------------------------------------------------------------------

% LOAD EUROAREA DATA
[x,xlag,T,n,k,g,ng,dates,names,namesg,tcode] = load_data_oil(inflindx,outpindx,standar,r,p,interF,quant,dfm); 

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
fpca      = zeros(T,r);  fqfa      = zeros(T,r,nq);
disp('Extracting PCA and VBQFA factors...')
for ifac = 1:r
   fpca(:,ifac) = extract(zscore(x(:,(ifac-1)*16+1:ifac*16)),1);
   fqfa(:,ifac,:) = VBQFA(zscore(x(:,(ifac-1)*16+1:ifac*16)),1,500,quant,0,1);
end
F(:,1:r*nq) = fqfa(:,:);
clc;

if AR1x == 0; ARindex = []; else, ARindex=1; end

% Storage space for Gibbs draws
F_draws     = zeros(T,r*nq,nsave/nthin);
L_draws     = zeros(r+AR1x+interX+ng,n,nq,nsave/nthin);
Phi_draws   = zeros(k,r*nq+ng,nsave/nthin);
Sigma_draws = zeros(n,nq,nsave/nthin);
OMEGA_draws = zeros(r*nq+ng,r*nq+ng,T,nsave/nthin);
z_draws     = zeros(T,n,nq,nsave/nthin); 

firf_save   = zeros(nhor,r*nq+ng,r*nq+ng,nsave/nthin);
yirf_save   = zeros(nhor,n*nq,r*nq+ng,nsave/nthin);
%% ============================| START MCMC |==============================
format bank;
fprintf('Now you are running QFAVAR with MCMC')
fprintf('\n')
fprintf('Iteration 000000')
savedraw = 0; tic;
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
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
            select = [1:interX+AR1x + incldg*ng, interX+AR1x+ng+floor((i-1)/16+1)];
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
        for ir = 1:r
            L(interX+AR1x+ng+ir,(ir-1)*16+1:ir*16,q) = L(interX+AR1x+ng+ir,(ir-1)*16+1:ir*16,q)./L(interX+AR1x+ng+ir,(ir-1)*16+q,q);
        end

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
    
    %% =====| Step 4: Sample factors [F;g]
    [F] = FFBS([xbar(:,:) g],Lfull,intF,Phic,[Sigma(:);zeros(ng,1)],Omegac,(r*nq+ng));    % Forward sampling backward smoothing algorithm   
    % Make sure factors are sign rotated, so that the same factor in different quantiles has the same interpretation
    % (factors are not sign identified between different quantile levels)
    for q = 1:nq
        for ir = 1:r
            Ctemp =  corrcoef([F(:,(q-1)*r+ir) fqfa(:,ir,q)]);
            F(:,(q-1)*r+ir) =  F(:,(q-1)*r+ir).*sign(Ctemp(1,2));
            L(interX+AR1x+ng+ir,(ir-1)*16+1:ir*16,q) = L(interX+AR1x+ng+ir,(ir-1)*16+1:ir*16,q).*sign(Ctemp(1,2));
        end
    end
    
    %%%%%%%% ================================================================================================== %%%%%%%%
    %%%%%%%% ================================| State equation (VAR dynamics) |================================= %%%%%%%%
    Flag = mlag2([F(:,1:nq*r),g],p);                 % Lags of factors for VAR part (DFM state equation)
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

        %% =======================================================================================================
        %% =================================| Structural inference (IRFs) |=======================================
        %% =====| 1) Generalized IRFs state equation (responses of quantile factors)                         
        ar_lags = Phi(interF+1:end,:)';
        ar0 = {ar_lags(:,1:r*nq+ng)};
        if p>1       
            for i = 2:p
                ar0 = [ar0 ar_lags(:,(i-1)*(r*nq+ng)+1:i*(r*nq+ng))];
            end
        end
        [firf] = armairf(ar0,[],'InnovCov',squeeze(OMEGA(:,:,end)),'Method','generalized','NumObs',nhor);
        firf = permute(firf,[1,3,2]);
        
        %%  =====| 2) GIRFs measurement equation (map IRFs to macro panel)
        nshocks = r*nq+ng;        
        yirf  = zeros(nhor+AR1x,n*nq,nshocks);
        LL    = zeros(n*nq,r*nq+ng);

        % stack loadings
        for q = 1:nq
            for i = 1:n
                LL((q-1)*n+1:q*n,1:ng) = L(interX+AR1x+1:interX+AR1x+ng,:,q)';
                LL((q-1)*n+1:q*n,(q-1)*r+1+ng:q*r+ng) = L(interX+AR1x+ng+1:end,:,q)';
            end
        end
                
        if AR1x == 1
           for j = 1:nshocks
               for h = 2:nhor+AR1x
                   yirf(h,:,j) = [firf(h-1,r*nq+1:end,j), firf(h-1,1:r*nq,j)]*LL(:,:)' +  yirf(h-1,:,j).*L(interX+AR1x,:);
               end
           end
           yirf = yirf(2:end,:,:);
        else
            for j = 1:nshocks
                yirf(:,:,j) = [firf(:,r*nq+1:end,j), firf(:,1:r*nq,j)]*LL(:,:)';
            end
        end
        %% save all GIRFs
        firf_save(:,:,:,savedraw) = firf;
        yirf_save(:,:,:,savedraw) = yirf;
    end
        
end

%% =====================================| PLOTS |==============================================
% 1) Plot factor estimates
F = squeeze(mean(F_draws,3));
plot_names = reshape(extractBefore(names,'.'),16,r);
ddates = datetime(dates,'InputFormat','yyyyMMM');

figure;
for i = 1:r
   subplot(round(r/2),2,i)
   plot(ddates,([F(:,i) F(:,i+r) F(:,i+2*r)]),'LineWidth',2)
   grid on
   legend({'0.10 Factor','0.50 Factor','0.90 Factor'})
   title(plot_names(1,i))
end

% 2) plot IRF of factors
figure;
for j = 1:ng
    varshock = r*nq + j;      
    irfarray = squeeze(firf_save(:,:,varshock,:));
    fnames = extractBefore(names([1:n/r:n],:),'.');
    for i = 1:r+1
        subplot(ng,r+1,(j-1)*(r+1) + i)
        if i <= r
            plot(1:nhor,mean(irfarray(:,i:r:r*nq,:),3),'LineWidth',2)
            legend({'10%','50%','90%'})
            title(fnames(i,1))
        else
            plot(1:nhor,mean(irfarray(:,r*nq+j,:),3),'LineWidth',2)
            title(namesg(j))
        end        
        if i == 1
            ylabel(namesg(j));
        end
    end
end

% Version 2
lev_ = [0.1, 0.5, 0.9];
for pp = 1:length(lev_)
    figure;
    for j = 1:ng
        varshock = r*nq + j;      
        irfarray = squeeze(firf_save(:,:,varshock,:));
        fnames = extractBefore(names([1:n/r:n],:),'.');
        for i = 1:r+1
            subplot(ng,r+1,(j-1)*(r+1) + i)
            if i <= r
                plot(1:nhor,mean(irfarray(:,i+(pp-1)*r,:),3),'LineWidth',2)              
                hold on
                shade(1:nhor,prctile(irfarray(:,i+(pp-1)*r,:),25,3),'w',1:nhor,prctile(irfarray(:,i+(pp-1)*r,:),75,3),'w',...
                'FillType',[2 1],'FillColor',{'black'},'FillAlpha',0.2,'LineStyle',"None")
                plot(1:nhor,zeros(1,nhor),'r')
                hold off
                legend({[string(lev_(pp)) + '%']})
                title(fnames(i,1))
            else
                plot(1:nhor,mean(irfarray(:,r*nq+j,:),3),'LineWidth',2)
                title(namesg(j))
            end        
            if i == 1
                ylabel(namesg(j));
            end
        end
    end
end

% version 3: "utter madness"

FigH = figure('Position', get(0, 'Screensize'));
for j = 1:ng
    varshock = r*nq + j;      
    irfarray = squeeze(firf_save(:,:,varshock,:));
    fnames = extractBefore(names([1:n/r:n],:),'.');
    for i = 1:r+1
        subplot(ng,r+1,(j-1)*(r+1) + i)
        if i <= r
            plot(1:nhor,mean(irfarray(:,i,:),3),'LineWidth',2)
            hold on
            shade(1:nhor,prctile(irfarray(:,i,:),25,3),'w',1:nhor,prctile(irfarray(:,i,:),75,3),'w',...
            'FillType',[2 1],'FillColor',[0 0.4470 0.7410],'FillAlpha',0.3,'LineStyle',"None")
            plot(1:nhor,mean(irfarray(:,i+r,:),3),'LineWidth',2,'Color',[0.6350 0.0780 0.1840])
            shade(1:nhor,prctile(irfarray(:,i+r,:),25,3),'w',1:nhor,prctile(irfarray(:,i+r,:),75,3),'w',...
            'FillType',[2 1],'FillColor',[0.6350 0.0780 0.1840],'FillAlpha',0.3,'LineStyle',"None")
            plot(1:nhor,mean(irfarray(:,i+2*r,:),3),'LineWidth',2,'Color',[0.9290 0.6940 0.1250])
            shade(1:nhor,prctile(irfarray(:,i+2*r,:),25,3),'w',1:nhor,prctile(irfarray(:,i+2*r,:),75,3),'w',...
            'FillType',[2 1],'FillColor',[0.9290 0.6940 0.1250],'FillAlpha',0.3,'LineStyle',"None")
            plot(1:nhor,zeros(1,nhor),'r')
            hold off
            %legend({'10%','50%','90%'})
            title(fnames(i,1))
        else
            plot(1:nhor,mean(irfarray(:,r*nq+j,:),3),'LineWidth',2)
            title(namesg(j))
        end        
        if i == 1
            ylabel(namesg(j));
        end
    end
end
%saveas(FigH, 'QAVAR_IRF_state_eq.jpg','jpeg');

% 3) plot IRF of panel
for ii = 1:size(namesg,2)
    varshock = r*nq + ii;
    countries = reshape(extractAfter(names,'.'),16,r);
    vars = extractBefore(names,'.');
    irfarray = squeeze(yirf_save(:,:,varshock,:));
    pgrid = reshape(1:r*n/r,n/r,r)';
    
    figure;
    for i = 1:n    
        subplot(r,n/r,i)
        if ii>2
            plot(mean(irfarray(:,i+n*2,:),3),'LineWidth',2)
            hold on
            plot(mean(irfarray(:,i+n,:),3),'LineWidth',2)
            plot(mean(irfarray(:,i,:),3),'LineWidth',2)
        else
            plot(mean(irfarray(:,i,:),3),'LineWidth',2)
            hold on
            plot(mean(irfarray(:,i+n,:),3),'LineWidth',2)
            plot(mean(irfarray(:,i+n*2,:),3),'LineWidth',2)
        end
        hold off
        grid on;
        if i<=n/r
            title(countries(i,1))
        end
        if sum(i==pgrid(:,1))==1
            ylabel(vars(i))
            legend({'10%','50%','90%'})
        end
    end
    sgtitle(namesg(ii)) 
end


for ii = 1:size(namesg,2)
    varshock = r*nq + ii;
    countries = reshape(extractAfter(names,'.'),16,r);
    vars = extractBefore(names,'.');
    irfarray = squeeze(yirf_save(:,:,varshock,:));
    pgrid = reshape(1:r*n/r,n/r,r)';
    
    figure;
    for i = 1:n    
        subplot(r,n/r,i)
            plot(1:nhor,mean(irfarray(:,i,:),3),'LineWidth',2)
            hold on
            shade(1:nhor,prctile(irfarray(:,i,:),25,3),'w',1:nhor,prctile(irfarray(:,i,:),75,3),'w',...
            'FillType',[2 1],'FillColor',[0 0.4470 0.7410],'FillAlpha',0.3,'LineStyle',"None")
            plot(1:nhor,mean(irfarray(:,i+n,:),3),'LineWidth',2,'Color',[0.6350 0.0780 0.1840])
            shade(1:nhor,prctile(irfarray(:,i+n,:),25,3),'w',1:nhor,prctile(irfarray(:,i+n,:),75,3),'w',...
            'FillType',[2 1],'FillColor',[0.6350 0.0780 0.1840],'FillAlpha',0.3,'LineStyle',"None")
            plot(1:nhor,mean(irfarray(:,i+2*n,:),3),'LineWidth',2,'Color',[0.9290 0.6940 0.1250])
            shade(1:nhor,prctile(irfarray(:,i+2*n,:),25,3),'w',1:nhor,prctile(irfarray(:,i+2*n,:),75,3),'w',...
            'FillType',[2 1],'FillColor',[0.9290 0.6940 0.1250],'FillAlpha',0.3,'LineStyle',"None")
            plot(1:nhor,zeros(1,nhor),'r')
            hold off
        %end
        hold off
        grid on;
        if i<=n/r
            title(countries(i,1))
        end
        if sum(i==pgrid(:,1))==1
            ylabel(vars(i))
            legend({'10%','50%','90%'})
        end
    end
    sgtitle(namesg(ii)) 
end

%%%%%%%%%%% FEVD

sirf_save = (firf_save.^2);
csirf_save = cumsum(sirf_save,1);
fevd_save = csirf_save./sum(csirf_save,3);
fevd = squeeze(mean(fevd_save,4));

fnames = {'CPI.10';'IP.10';'CPI.50';'IP.50';'CPI.90';'IP.90'};
labels = {'oil-prod';'IGREA';'loil-price';'GSCPI';'GEPU'};

j = 0;
for q = 1:nq
    j = j+1;
    subplot(nq,2,j)
    area(squeeze(fevd(:,1+(q-1)*r,nq*r+1:end)),'FaceColor','flat');
    xlim([1 nhor]); ylim([0 1]); grid on;
    title(['Forecast error decomposition of ' cellstr(fnames(j))])
    legend(labels);
    j = j+1;
    subplot(nq,2,j)
    area(squeeze(fevd(:,2+(q-1)*r,nq*r+1:end)),'FaceColor','flat');
    xlim([1 nhor]); ylim([0 1]); grid on;
    title(['Forecast error decomposition of ' cellstr(fnames(j))])
    legend(labels);
end

%% save results
save(sprintf('%s.mat','QFAVAR_GIRFs'),'-mat');
