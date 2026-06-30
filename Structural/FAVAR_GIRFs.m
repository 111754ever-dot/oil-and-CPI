%% FAVAR_GIRFs.m    Bayesian Factor-Augmented VAR Model
%%                  (MCMC estimation + Generalized IRFs)
%==========================================================================
%  The model is of the form
%          _       _        _           _    _        _     _    _
%         |  x_{t}  |      |  L       G  |  |F_{t}(tau)|   | e_{t} |
%         |         |  =   |             |  |          | + |       |
%         |_ g_{t} _|      |_ 0       I _|  |_  g_{t} _|   |_  0  _|
%         
%         _          _           _            _
%        | F_{t}(tau) |         | F_{t-1}(tau) |  
%        |            |  =  Phi |              |  + u_{t},
%        |_   g_{t}  _|         |_  g_{t-1}   _|
%
%  where e_{t} ~ N(0, Sigma) and u_{t} ~ N(0, Q), Sigma and Q are covariance
%  matrices.
% =========================================================================
% Written by: 
%
%      Dimitris Korobilis         and       Maximilian Schroeder
%    University of Glasgow        and   Norwegian BI Business School
%
% First version: 06 July 2022
% This version: 11 March 2023
% =========================================================================

clear all; clc; close all;
tic;

% Add path of additional folders
addpath('functions')
addpath('data')
%---------------------------| USER INPUT |---------------------------------
% Model settings
r         = 5;              % Number of factors
p         = 2;              % Number of lags
interX    = 0;              % Intercept in measurement equation
interF    = 0;              % Intercept in state equation
AR1x      = 0;              % Include own lag
incldg    = 1;              % Include global factors g in the measurement equation
dfm       = 0;              % 0: estimate FAVAR; 1: estimate DFM
var_sv    = 0;              % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 
standar   = 0;              % 0: no standardization; 1: standardize only x variables; 2: standardize both x and g variables
inflindx  = 1;              % 1: HICP Total; 2: HICP less energy; 3: HICP less food and energy
outpindx  = 2;              % 1: Unemployment; 2: Industrial Production (OECD data)
nhor      = 60;             % Horizon for IRFs, FEVDs, connectedness

% Gibbs-related preliminaries
nsave     = 100000;         % Number of draws to store
nburn     = 5000;           % Number of draws to discard
ngibbs    = nsave + nburn;  % Number of total draws
nthin     = 50;             % Save every nthin draw to reduce MCMC correlation
iter      = 100;            % Print every "iter" iteration
%--------------------------------------------------------------------------
% LOAD EUROAREA DATA
[x,xlag,T,n,k,g,ng,dates,names,namesg,tcode] = load_data(inflindx,outpindx,standar,r,p,interF,[],dfm);

%=========| Estimation
% Horseshoe prior for L
lambdaL   = 0.1*ones(n,r+interX+ng);          % "local" shrinkage parameters
tauL      = 0.1*ones(n,1);                    % "global" shrinkage parameter
nuL       = 0.1*ones(n,r+interX+ng);  
xiL       = 0.1*ones(n,1); 

% Horseshoe prior for Phi
lambdaPhi = 0.1*ones(r+ng,k);                 % "local" shrinkage parameters
tauPhi    = 0.1*ones(r+ng,1);                 % "global" shrinkage parameter
nuPhi     = 0.1*ones(r+ng,k);  
xiPhi     = 0.1*ones(r+ng,1); 

% Choose sampling algorithm for VAR parameters
est_meth = 1 + double(k>T);

% Initialize matrices
xbar      = 0*x;
Lbar      = zeros(n,r,T);
Lbar2     = zeros(n,ng,T);
L         = zeros(r+interX+ng,n);
Sigma     = 0.1*ones(n,1);
Phi       = 0.1*ones(k,(r+ng));
Omega     = 0.1*ones(1,r+ng);
Omega_t   = 0.1*ones(T-p,r+ng);
OMEGA     = 0.1*ones(r+ng,r+ng,T);
h         = 0.1*ones(T-p,r+ng);   
sig       = 0.1*ones(r+ng,1);
FL        = zeros(T,n);  
Omegac    = zeros((r+ng)*p,(r+ng)*p,T);
Phic      = [Phi(interF+1:end,:)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1),r+ng)]; 
Omegac(1:r+ng,1:r+ng,:) = repmat(diag(Omega),1,1,T); 
QL        = 1*ones(r+interX+ng,n);
QPhi      = 1*ones(k,(r+ng));
intF      = zeros(T,(r+ng)*p);

disp('Estimating QPCA factors...')
fpca      = zeros(T,r);
for ifac = 1:r
    fpca(:,ifac) = extract(zscore(x(:,(ifac-1)*9+1:ifac*9)),1);
end
F = fpca;
clc;

if AR1x == 0; ARindex = []; else, ARindex=1; end

% Storage space for Gibbs draws
F_draws     = zeros(T,r,nsave/nthin);
L_draws     = zeros(r+interX+ng,n,nsave/nthin);
Phi_draws   = zeros(k,r+ng,nsave/nthin);
Sigma_draws = zeros(n,nsave/nthin);
OMEGA_draws = zeros(r+ng,r+ng,T,nsave/nthin);

firf_save   = zeros(nhor,r+ng,r+ng,nsave/nthin);
yirf_save   = zeros(nhor,n,r+ng,nsave/nthin);

%% ============================| START MCMC |==============================
format bank;
fprintf('Now you are running FAVAR with MCMC')
fprintf('\n')
fprintf('Iteration 000000')
savedraw = 0;
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%6d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end

    %%%%%%%% ================================================================================================== %%%%%%%%
    %%%%%%%% =====================| QFAVAR measurement equation (Factor extraction) |========================== %%%%%%%%
    Lc    = zeros(n,r+ng,T);
    Lfull = zeros(n+ng,r+ng,T);
    x_tilde = 0*x;   
    for i = 1:n  % equation-by-equation estimation
        %% Select the factors that correspond to the i-th variable           
        F_all = [ones(T,(interX==1)), xlag(:,ARindex*i), g, F(:,1:r)];
        select = [1:interX+AR1x + incldg*ng, interX+AR1x+ng+floor((i-1)/9+1)];
        F_select = F_all(:,select);

        %% =====| Step 1: Sample loadings L
        x_tilde(:,i) = x(:,i)./sqrt(Sigma(i));                                             % Standardized LHS variables
        F_tilde = F_select./sqrt(Sigma(i));                                                                     % Standardized RHS variables                     
        L(select,i) = randn_gibbs(x_tilde(:,i),F_tilde,QL(select,i),T,length(select),1);   % Sample loadings Lambda       
        [QL(select,i),~,~,lambdaL(i,select),tauL(i),nuL(i,select),xiL(i)] = ...
            horseshoe_prior(L(select,i)',length(select),tauL(i),nuL(i,select),xiL(i));     % sample prior variance of loadings Lambda        
        %Normalization restriction
        if mod(i,9)==3;  L(interX+AR1x+ng+floor((i-1)/9+1),i) = 1; end

        %% =====| Step 3: Sample factor regression variances Sigma
        FL(:,i) = F_all*L(:,i);       
        a1 = 0.01 + T/2;   sse = (x(:,i) - FL(:,i)).^2;
        a2 = 0.01 + sum(sse./2);       
        Sigma(i,1) = 1./gamrnd(a1,1./a2);                                                  % Sample Sigma from inverse-Gamma

        % Remove intercept from x
        Ftemp = [ones(T,(interX==1)), xlag(:,ARindex*i)];
        xbar(:,i) = (x(:,i) - Ftemp*L(1:interX+AR1x,i));
    end
    
    % Normalize loadings
    for ir = 1:r
        L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9) = L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9)./L(interX+AR1x+ng+ir,(ir-1)*9+1);
    end

    % In the QFAVAR, Lc also has the last ng columns corresponding to variables g_{t}
    for t = 1:T      
        Lfull(:,:,t) = [[L(interX+ng+1:end,:)', L(1:interX + ng,:)']; zeros(ng,r) eye(ng)];
    end
    
    %% =====| Step 4: Sample factors [F;y]
   [F] = FFBS([xbar, g],Lfull,intF,Phic,[Sigma(:);zeros(ng,1)],Omegac,(r+ng));    % Forward sampling backward smoothing algorithm
    % Make sure factors are sign rotated, in order to conform with their different quantile levels
    for ir = 1:r
        Ctemp =  corrcoef([F(:,ir) fpca(:,ir)]);
        F(:,ir) =  F(:,ir).*sign(Ctemp(1,2));
        L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9) = L(interX+AR1x+ng+ir,(ir-1)*9+1:ir*9).*sign(Ctemp(1,2));
    end
    %%%%%%%% ================================================================================================== %%%%%%%%
    %%%%%%%% ================================| State equation (VAR dynamics) |================================= %%%%%%%%
    Flag = mlag2([F(:,1:r),g],p);                               % Lags of factors for VAR part (DFM state equation)
    Fy = [F(p+1:end,1:r), g(p+1:end,:)];          % LHS variables for state equation (correct observations for number of lags)
    Fx = [ones(T-p,0+interF) Flag(p+1:end,:)];       % RHS variables for state equation (correct observations for number of lags)
    resid = zeros(T-p,r+ng);
    A_ = eye(r+ng);
    
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
        F_draws(:,:,savedraw)       = F(:,1:r);
        L_draws(:,:,savedraw)       = L;
        Phi_draws(:,:,savedraw)     = Phi;
        Sigma_draws(:,savedraw)     = Sigma;
        OMEGA_draws(:,:,:,savedraw) = OMEGA;
        
        %% =======================================================================================================
        %% =================================| Structural inference (IRFs) |=======================================
        %% =====| 1) Generalized IRFs state equation (responses of quantile factors)                         
        ar_lags = Phi(interF+1:end,:)';
        ar0 = {ar_lags(:,1:r+ng)};
        if p>1       
            for i = 2:p
                ar0 = [ar0 ar_lags(:,(i-1)*(r+ng)+1:i*(r+ng))];
            end
        end
        [firf] = armairf(ar0,[],'InnovCov',squeeze(OMEGA(:,:,end)),'Method','generalized','NumObs',nhor);
        firf = permute(firf,[1,3,2]);
        
        %%  =====| 2) GIRFs measurement equation (map IRFs to macro panel)
        nshocks = r+ng;        
        yirf = zeros(nhor+AR1x,n,nshocks);
        LL    = zeros(n,r+ng);

        % stack loadings
        for i = 1:n
            LL(:,1:ng) = L(interX+AR1x+1:interX+AR1x+ng,:)';
            LL(:,1+ng:r+ng) = L(interX+AR1x+ng+1:end,:)';
        end
                
        if AR1x == 1
           for j = 1:nshocks
               for h = 2:nhor+AR1x
                   yirf(h,:,j) = [firf(h-1,r+1:end,j), firf(h-1,1:r,j)]*LL(:,:)' +  yirf(h-1,:,j).*L(interX+AR1x,:);
               end
           end
           yirf = yirf(2:end,:,:);
        else
            for j = 1:nshocks
                yirf(:,:,j) = [firf(:,r+1:end,j), firf(:,1:r,j)]*LL(:,:)';
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

figure;
for i = 1:r
   subplot(round(r/2),2,i)
   plot(F(:,i),'LineWidth',2)
   grid on   
end



FigH = figure('Position', get(0, 'Screensize'));
for j = 1:ng
    varshock = r + j;      
    irfarray = squeeze(firf_save(:,:,varshock,:));
    fnames = extractBefore(names([1:n/r:n],:),'.');
    for i = 1:r+1
        subplot(ng,r+1,(j-1)*(r+1) + i)
        if i <= r
            plot(1:nhor,mean(irfarray(:,i,:),3),'LineWidth',2)
            hold on
            shade(1:nhor,prctile(irfarray(:,i,:),25,3),'w',1:nhor,prctile(irfarray(:,i,:),75,3),'w',...
                'FillType',[2 1],'FillColor',{'black'},'FillAlpha',0.2,'LineStyle',"None")
            plot(1:nhor,zeros(1,nhor),'r')
            hold off
            title(fnames(i,1))
        else
            plot(1:nhor,mean(irfarray(:,r+j,:),3),'LineWidth',2)
            title(namesg(j))
        end        
        if i == 1
            ylabel(namesg(j));
        end
    end
end
saveas(FigH, 'FAVAR_IRF_state_eq.jpg','jpeg');


% 3) plot IRF of panel
for ii = 1:size(namesg,2)
    varshock = r + ii;
    countries = reshape(extractAfter(names,'.'),9,r);
    vars = extractBefore(names,'.');
    irfarray = squeeze(yirf_save(:,:,varshock,:));
    pgrid = reshape(1:r*n/r,n/r,r)';
    
    figure;
    for i = 1:n    
        subplot(r,n/r,i)
        plot(mean(irfarray(:,i,:),3),'LineWidth',2)
        hold on
        shade(1:nhor,prctile(irfarray(:,i,:),25,3),'w',1:nhor,prctile(irfarray(:,i,:),75,3),'w',...
                'FillType',[2 1],'FillColor',{'black'},'FillAlpha',0.2,'LineStyle',"None")
        plot(1:nhor,zeros(1,nhor),'r')
        hold off
        grid on;
        if i<=n/r
            title(countries(i,1))
        end
        if sum(i==pgrid(:,1))==1
            ylabel(vars(i))
        end
    end
    sgtitle(namesg(ii)) 
end

%% save results
save(sprintf('%s.mat','FAVAR_GIRFs'),'-mat');