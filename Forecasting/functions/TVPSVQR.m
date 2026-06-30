function [yfore] = TVPSVQR(y,X,Xfore,nsave,nburn,nthin,quant,tvp_reg,sv_reg,prior)
%% tvpsvqr: Function that does flexible estimation of Bayesian quantile regression
%  INPUTS
%    y      LHS variable of interest, measured for t=h+1:T, h is forecast horizon
%    X      RHS matrix (typically intercept, lags of y and lagged exogenous predictors, measured for t=1:T-h)
%  Xfore    Vector of values of X at time T (to forecast y_{T+h})
%  nsave    Number of Gibbs draws to store after convergence
%  nburn    Number of initial Gibbs draws to discard
%  quant    Vector of quantiles of y to estimate
% tvp_reg   0: Constant parameter regression; 1: TVP regression; 2: TVP only on intercept
% sv_reg    0: Constant variance; 1: Stochastic volatility
%  prior    Shrinkage prior to use. Choices are:
%                                   1: Normal-iGamma prior (Student t)
%                                   2: SSVS with Normal(0,tau_0) and Normal-iGamma components
%                                   3: Horseshoe prior
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% First version: 30 October 2020
% This version: 30 October 2020
%==========================================================================

[T,p] = size(X);

% ==============| create matrices useful for TVP estimation  
if tvp_reg == 0 % Constant parameter regression case
    x = X;
    Tp = p;
    model = 'CP';
elseif tvp_reg == 1 % TVP regression case
    % We need to create an expanded matrix x which has a lower triangular
    % structure with T rows and Tp columns for the TVPs. We also add one column
    % of ones for intercept (these are time-invariant).
    Tp = T*p;   
    H = speye(Tp,Tp) - sparse(p+1:Tp,1:(T-1)*p,ones(1,(T-1)*p),Tp,Tp);
    Hinv = speye(Tp,Tp)/H;
    x = full(SURform(X)*Hinv);
    model = 'TVP';
elseif tvp_reg == 2 % TVP only on intercept case
    Tp = T+(p-1);
    x = [tril(ones(T)), X(:,2:end)];
    model = 'TVI';
end

% ==============| Define quantiles
%quant = [.05,0.1,0.25,0.5,0.75,0.9,.95];   % Specify the quantiles to estimate
n_q = length(quant);                       % Number of quantiles
% The next three quantities are required for the posterior of latent quantities z (see Gibbs sampling Step 1)
tau_sq = 2./(quant.*(1-quant));
theta = (1-2*quant)./(quant.*(1-quant));
theta_sq = theta.^2;

% Quantity to use in calculation of the auxiliary quantiles
%C = tril(log(quant./quant')./(1-repmat(quant,n_q,1))' ) - triu(log((1-quant)./(1-quant'))./(repmat(quant,n_q,1))' );
for i = 1:n_q; for j = 1:n_q
    tau = quant(i);
    pp = quant(j);
    if pp>=tau       
        C(j,i) = log(tau/pp)/(1-pp);
    elseif pp<tau
        C(j,i) = -log((1-tau)/(1-pp))/pp;
    end;  end
end
sigma_k = 100;  % Use similar value to T. Rodrigues & Y. Fan (2017), Journal of Computational and Graphical Statistics

% ==============| Define priors
% prior for beta
Q    = .01*ones(Tp,n_q);
miu  = zeros(Tp,n_q);  
if prior == 1
    % Student_T shrinkage prior
    b0 = 0.01;
elseif prior == 2       
    % SSVS prior
    c0 = (0.01)^2;
    tau = 4*ones(Tp,n_q);
    pi0 = 0.25;
    b0 = 0.01;
elseif prior == 3   
    % Horseshoe prior
    lambda = 0.1*ones(Tp,n_q);     % "local" shrinkage parameters, one for each Tp+1 parameter element of betaD
    tau = 0.1*ones(1,n_q);           % "global" shrinkage parameter for the whole vector betaD       
    nu = 0.1*ones(Tp,n_q);  
    xi = 0.1*ones(1,n_q);  
end

% ==============| Initialize vectors/matrices
beta = rand(Tp,n_q);
betaD = rand(Tp,n_q);
beta_mat = zeros(T,p,n_q);
sigma_t = 0.1*ones(T,n_q);
h = ones(T,n_q); 
sig = 0.1*ones(n_q,1);

% Storage space for Gibbs draws (be careful these vectors can require too
% much memory if you overdo it with either n_q or nsave, or if you insert
% e.g. daily data with large T)
beta_draws  = zeros(T,p,n_q,nsave/nthin);
sigma_draws = zeros(T,n_q,nsave/nthin);
z_draws     = zeros(T,n_q,nsave/nthin);
yfore       = zeros(n_q,nsave/nthin);

%% =========| GIBBS sampler starts here
iter = 500;             % Print every "iter" iteration
savedraw = 0;
if sv_reg == 1
    model2 = 'SV';
elseif sv_reg == 0
    model2 = 'CV';
end
fprintf('\n')
fprintf(['Model: QR with ' str2mat(model) ' and ' str2mat(model2)])
fprintf('\n')
fprintf('Iteration 0000')
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end   
    
    % =====| Step 1: Sample latent indicators z
    % Note: I sample here using MATLAB vectorized operations, instead of
    % inefficient "for" loops. Variable z is T x n_q, i.e. one time time
    % series corresponding for each quantile we estimate
    k1 = sqrt(theta_sq + 2*tau_sq)./abs(y - x*betaD);         % posterior moment k1 of z
    k2 = (theta_sq + 2*tau_sq)./(sigma_t.*tau_sq);            % posterior moment k2 of z
    z  = min(1./random('InverseGaussian',k2,k1,T,n_q),1e+6); % Sample z from Inverse Gaussian
    
    % =====| Step 2: Sample regression variance
    if sv_reg == 1   % Sample Stochastic Volatility
        yhat = (y - x*betaD - theta.*z)./sqrt(tau_sq.*z);        % regression residuals
        ystar  = log(yhat.^2 + 1e-6);                           % log squared residuals
        for q = 1:n_q       
            [h(:,q), ~] = SVRW(ystar(:,q),h(:,q),sig(q,1),1);    % log stochastic volatility using Chan's filter   
        end
        sigma_t  = exp(h);                                       % convert log-volatilities to variances
        r1 = 10 + T - 1;   r2 = 0.001 + sum(diff(h).^2)';         % posterior moments of variance of log-volatilities
        sig = 1./gamrnd(r1./2,2./r2);                            % sample variance of log-volatilities from Inverse Gamma
    elseif sv_reg == 0   % Sample constant regression variance
        a1 = 0.1 + 3*T/2;    sse = (y - x*betaD - theta.*z).^2;
        a2 = 0.1 + sum(sse./(2*z.*tau_sq)) + sum(z);       
        sigma2 = 1./gamrnd(a1,1./a2);                            % Sample from inverse-Gamma
        sigma_t = repmat(sigma2,T,1);                            % Convert to a T x n_q matrix (but replicate same value for all t=1,...,T)
    end
    
    % =====| Step 3: Sample regression coefficients beta_t
    u = (1./sqrt(sigma_t.*tau_sq.*z));                       % This is the standard deviation of the Asymetric Laplace errors
    y_tilde = (y - theta.*z).*u;                             % Take the residual and standardize
    for q = 1:n_q
        xnew = x.*u(:,q);                                    % Also standardize x           
        betaD(:,q) = randn_gibbs(y_tilde(:,q),xnew,Q(:,q),T,Tp,1);  % sample regression coefficients(Fy_tilde,FX_tilde,[QPhi(:,i);9*ones(i-1,1)],T-p,k+i-1,est_meth)
        if tvp_reg == 1
            beta(:,q) = Hinv*betaD(:,q);                % betaD are TVPs in differences (beta_{t} - beta_{t-1}), convert to vector of TVPs in levels (beta_{t})
            beta_mat(:,:,q) = reshape(beta(:,q),p,T)';  % beta_mat is T x (p+1) x n_q matrix of TVPs
        elseif tvp_reg == 2
            beta(:,q) = [cumsum(betaD(1:T,q));betaD(T+1:end,q)];
            beta_mat(:,:,q) = [beta(1:T,q) , repmat(betaD(T+1:end,q)',T,1)];
        elseif tvp_reg == 0
            beta(:,q) = betaD(:,q);                     % In the constant parameter case, betaD are the beta coefficients we need
            beta_mat(:,:,q) = repmat(beta(:,q)',T,1);
        end
        
        %--draw prior variance
        if prior == 1
            [Q(:,q),~,miu(:,q)] = student_T_prior(betaD(:,q),b0);
        elseif prior == 2
            [Q(:,q),~,miu(:,q),tau(:,q)] = ssvs_prior(betaD(:,q),c0,tau(:,q),b0,pi0);
        elseif prior == 3
            [Q(:,q),~,miu(:,q),lambda(:,q),tau(:,q),nu(:,q),xi(:,q)] = horseshoe_prior(betaD(:,q),Tp,tau(:,q),nu(:,q),xi(:,q));
            %                                                        horseshoe_prior(L(select,i,q)',length(select),tauL(i,q),nuL(i,select,q),xiL(i,q)); 
        end                                                           
    end
        
    if irep > nburn && mod(irep,nthin)==0
        savedraw = savedraw + 1;
        % Save draws of parameters
        beta_draws(:,:,:,savedraw) = beta_mat;
        sigma_draws(:,:,savedraw)  = sigma_t;
        z_draws(:,:,savedraw)      = z;        
    
        % Forecast of quantiles from the QR model
        yfore(:,irep-nburn) = Xfore*squeeze(beta_mat(T,:,:));
    end  
end

%fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8)
end



