%% FORECASTING_1_vsFAVAR.m:
% First forecasting exercise in Korobilis and Schroeder (2023) Monitoring 
% Multicountry Macroeconomic Risk.
%
% Compare the benchmark QFAVAR with a benchmark FAVAR specification with identical 
% horseshoe priors (no SV or other features, just the benchmark specifications) 
% =========================================================================
% Written by: 
%
%      Dimitris Korobilis         and       Maximilian Schroeder
%    University of Glasgow        and   Norwegian BI Business School
%
% First version: 06 July 2022
% This version: 05 November 2023
% =========================================================================

% Reset everything, clear memory and screen
clear; close all; clc;
% Start clock
tic;

% Add path of random number generators
addpath('functions')
addpath('data')
 
%===========================| USER INPUT |=================================
% Model specification
quant = [10,50,90]./100;   % Specify the quantiles to estimate
q10 = find(quant==0.1);  qmed = find(quant==0.5);  q90 = find(quant==0.9);
nq = length(quant);        % Number of quantiles 

r = 5;                   % Number of factors
p = 2;                   % Number of lags
interX = 0;              % Intercept in factor equation
interF = 0;              % Intercept in VAR equation
AR1x   = 0;              % Include own lag
incldg    = 1;           % Include global factors g in the measurement equation
dfm       = 0;           % 0: estimate QFAVAR; 1: estimate QDFM
standar   = 2;           % 0: no standardization; 1: standardize only x variables; 2: standardize both x and g variables
ALsampler = 1;           % Asymmetric Laplace sampler, 1: Khare and Robert (2012); 2: Kozumi and Koyabashi (2011) 
var_sv    = 0;           % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 

% Forecasting
h          = 24;         % Number of prediction steps ahead
inflindx   = 1;          % 1: HICP Total; 2: HICP less energy; 3: HICP less food and energy
outpindx   = 2;          % 1: Unemployment; 2: Industrial Production (OECD data)

% MCMC setting (total iterations are nsave+nburn, stored iterations are nsave/nthin)
nsave = 2000;            % Store nsave MCMC iterations from posteriors
nburn = 500;            % Discard nburn iterations for convergence
nthin = 20;              % Save every nthin iteration to reduce correlation in the chain


%% ===========================| LOAD DATA |=================================

%% ======|Load series to extract factors from
[x,xlag,T,n,k,g,ng,dates,names,namesg,tcode] = load_data(inflindx,outpindx,standar,r,p,interF,quant,dfm);

% Start with  first_per% of observations for first period forecast
first_per = 0.5;
T_full    = T-h;
T_thres   = round(first_per*T_full); 
anumber   = T_full-T_thres+1;
varnum    = size(x,2);

%Generate Matrices for storing results
CQs      = zeros(anumber,nq,varnum,h,2);  % Save raw conditional quantiles   
QScore10 = zeros(anumber,varnum,h,2);     % Quantile score 10%
QScore90 = zeros(anumber,varnum,h,2);     % Quantile score 90%
PL       = zeros(anumber,varnum,h,2);     % Predictive likelihood (based on kernel smoothing)
PITS     = zeros(anumber,varnum,h,2);     % PITS (based on kernel smoothing)
PE       = zeros(anumber,varnum,h,2);     % Prediction Error   
MSPE     = zeros(anumber,varnum,h,2);     % Mean square prediction error
    
for sample = T_thres:T_full
    disp('  ')           
    fprintf(['<strong>YOU ARE RUNNING SAMPLE ' num2str(sample) ' OF ' num2str(T_full) '</strong> \n'] )    
    disp('  ')        
    
    xf_true = x(sample+1:sample+h,:);
        
    %% QFAVAR forecasts
    [xforeQR] = QFAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,k,g(1:sample,:),ng,r,p,quant,h,interX,interF,AR1x,incldg,var_sv,ALsampler,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants = xforeQR(ivar,:,nfore);   
            CQs(sample-T_thres+1,:,ivar,nfore,1)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,1)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,1)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,1)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,1)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,1)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,1)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end

    %% FAVAR forecasts
    [xforeMR] = FAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,(r+ng)*p+interF,g(1:sample,:),ng,r,p,h,interX,interF,AR1x,incldg,var_sv,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants = quantile(xforeMR(ivar,nfore,:),quant);   
            CQs(sample-T_thres+1,:,ivar,nfore,2)     = post_mean_quants';       
            QScore10(sample-T_thres+1,ivar,nfore,2)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,2)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,2)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,2)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,2)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,2)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end
end %recursive forecasts


%% Plot forecast densities in Dec 2021
quant = [5,10,25,50,75,90,95]./100;     % Specify the quantiles to estimate
T = T-12;
[xforeQR] = QFAVAR_FORE2(x(1:T,:),xlag(1:T,:),T,n,(r*length(quant)+ ng)*p + interF,g(1:T,:),ng,r,p,quant,h,interX,interF,AR1x,incldg,var_sv,ALsampler,2*nsave,nburn,nthin);
[xforeMR] = FAVAR_FORE2(x(1:T,:),xlag(1:T,:),T,n,(r+ng)*p+interF,g(1:T,:),ng,r,p,h,interX,interF,AR1x,incldg,var_sv,2*nsave,nburn,nthin);

figure
for i=1:9
    subplot(3,3,i)
    [f,xi] = ksdensity(xforeQR(i,:,12),'Bandwidth',0.7);
    plot(xi,f,'LineWidth',3);
    hold all
    post_mean_quants = quantile(xforeMR(i,12,:),quant);
    [f,xi] = ksdensity(post_mean_quants,'Bandwidth',0.7);
    plot(xi,f,'--r','LineWidth',3);
    yl = ylim;
    plot([x(T+12,i) x(T+12,i)],yl,'LineWidth',3)
    title(names(i))
    fontsize(gcf,18,'points')
end
sgtitle('1-step ahead predictive densities of inflation')

figure
for i=10:18    
    subplot(3,3,i-9)
    [f,xi] = ksdensity(xforeQR(i,:,12),'Bandwidth',0.7);
    plot(xi,f,'LineWidth',3);
    hold all
    post_mean_quants = quantile(xforeMR(i,12,:),quant);
    [f,xi] = ksdensity(post_mean_quants,'Bandwidth',0.7);
    plot(xi,f,'--r','LineWidth',3);
    yl = ylim;
    plot([x(T+12,i) x(T+12,i)],yl,'LineWidth',3)
    title(names(i))
    fontsize(gcf,18,'points')
end
sgtitle('1-step ahead predictive densities of industrial production')


%% Calculate ttests for Qscores
[H,P,CI,STATS] = ttest2(QScore10(:,:,:,1),QScore10(:,:,:,2));
t_stat10 = squeeze(STATS.tstat);
[H,P,CI,STATS] = ttest2(QScore90(:,:,:,1),QScore90(:,:,:,2));
t_stat90 = squeeze(STATS.tstat);

%% Save results to mat file
save(sprintf('%s.mat','FORECASTING_1_new_data'),'-mat');
