%% FORECASTING_0_QFAVAR.m:
% Zeroth forecasting exercise in Korobilis and Schroeder (2023) Monitoring 
% Multicountry Macroeconomic Risk. (exercise results not included in the paper)
%
% Compare the benchmark QFAVAR with alternative specifications (intercept
% in measurement and/or state equations, stochastic volatility in state equation).
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
incldg    = 1;           % Include global factors g in the measurement equation
AR1x      = 0;           % Include own lag
dfm       = 0;           % 0: estimate QFAVAR; 1: estimate QDFM
standar   = 2;           % 0: no standardization; 1: standardize only x variables; 2: standardize both x and g variables
ALsampler = 1;           % Asymmetric Laplace sampler, 1: Khare and Robert (2012); 2: Kozumi and Koyabashi (2011) 

% Forecasting
h          = 24;         % Number of prediction steps ahead
inflindx   = 1;          % 1: HICP Total; 2: HICP less energy; 3: HICP less food and energy
outpindx   = 2;          % 1: Unemployment; 2: Industrial Production (OECD data)

% MCMC setting (total iterations are nsave+nburn, stored iterations are nsave/nthin)
nsave = 2000;            % Store nsave MCMC iterations from posteriors
nburn = 500;             % Discard nburn iterations for convergence
nthin = 20;              % Save every nthin iteration to reduce correlation in the chain


%% ===========================| LOAD DATA |=================================

%% ======|Load series to extract factors from
[x,xlag,T,n,k,g,ng,dates,names,namesg,tcode] = load_data(inflindx,outpindx,standar,r,p,1,quant,dfm);

% Start with  first_per% of observations for first period forecast
first_per = 0.5;
T_full    = T-h;
T_thres   = round(first_per*T_full); 
anumber   = T_full-T_thres+1;
varnum    = size(x,2);

%Generate Matrices for storing results
CQs      = zeros(anumber,nq,varnum,h,4);  % Save raw conditional quantiles   
QScore10 = zeros(anumber,varnum,h,4);     % Quantile score 10%
QScore90 = zeros(anumber,varnum,h,4);     % Quantile score 90%
PL       = zeros(anumber,varnum,h,4);     % Predictive likelihood (based on kernel smoothing)
PITS     = zeros(anumber,varnum,h,4);     % PITS (based on kernel smoothing)
PE       = zeros(anumber,varnum,h,4);     % Prediction Error   
MSPE     = zeros(anumber,varnum,h,4);     % Mean square prediction error
    
for sample = T_thres:T_full
    disp('  ')           
    fprintf(['<strong>YOU ARE RUNNING SAMPLE ' num2str(sample) ' OF ' num2str(T_full) '</strong> \n'] )    
    disp('  ')        
    
    xf_true = x(sample+1:sample+h,:);
        
    %% QFAVAR1 forecasts (benchmark)
    interX = 0;              % Intercept in factor equation
    interF = 0;              % Intercept in VAR equation
    var_sv = 0;              % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 
    [xforeQR] = QFAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,(r*nq + ng)*p + interF,g(1:sample,:),ng,r,p,quant,h,interX,interF,AR1x,incldg,var_sv,ALsampler,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants                         = xforeQR(ivar,:,nfore);   
            CQs(sample-T_thres+1,:,ivar,nfore,1)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,1)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,1)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,1)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,1)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,1)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,1)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end

    %% QFAVAR2 forecasts (with SV)
    interX = 0;              % Intercept in factor equation
    interF = 0;              % Intercept in VAR equation
    var_sv = 1;              % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 
    [xforeQR] = QFAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,(r*nq + ng)*p + interF,g(1:sample,:),ng,r,p,quant,h,interX,interF,AR1x,incldg,var_sv,ALsampler,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants                         = xforeQR(ivar,:,nfore);   
            CQs(sample-T_thres+1,:,ivar,nfore,2)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,2)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,2)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,2)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,2)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,2)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,2)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end

    %% QFAVAR3 forecasts (with intercepts)
    interX = 1;              % Intercept in factor equation
    interF = 1;              % Intercept in VAR equation
    var_sv = 0;              % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 
    [xforeQR] = QFAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,(r*nq + ng)*p + interF,g(1:sample,:),ng,r,p,quant,h,interX,interF,AR1x,incldg,var_sv,ALsampler,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants                         = xforeQR(ivar,:,nfore);   
            CQs(sample-T_thres+1,:,ivar,nfore,3)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,3)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,3)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,3)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,3)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,3)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,3)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end

    %% QFAVAR4 forecasts (with intercepts and SV)
    interX = 1;              % Intercept in factor equation
    interF = 1;              % Intercept in VAR equation    
    var_sv = 1;              % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 
    [xforeQR] = QFAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,(r*nq + ng)*p + interF,g(1:sample,:),ng,r,p,quant,h,interX,interF,AR1x,incldg,var_sv,ALsampler,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants                         = xforeQR(ivar,:,nfore);   
            CQs(sample-T_thres+1,:,ivar,nfore,4)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,4)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,4)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,4)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,4)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,4)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,4)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end


end %recursive forecasts



pdates = dates(T_thres+h:T_full+h);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,1,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,1,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,1,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i,1,4))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,1,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,1,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,1,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i,1,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,1,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,1,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,1,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i+9,1,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,1,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,1,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,1,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i+9,1,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_0_h=1.jpg")

pdates = dates(T_thres+h:T_full+h);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,6,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,6,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,6,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i,6,4))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,6,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,6,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,6,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i,6,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,6,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,6,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,6,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i+9,6,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,6,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,6,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,6,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i+9,6,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_0_h=6.jpg")


pdates = dates(T_thres+h:T_full+h);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,12,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,12,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,12,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i,12,4))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,12,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,12,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,12,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i,12,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,12,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,12,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,12,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i+9,12,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,12,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,12,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,12,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i+9,12,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_0_h=12.jpg")


pdates = dates(T_thres+h:T_full+h);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,24,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,24,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,24,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i,24,4))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,24,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,24,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,24,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i,24,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,24,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,24,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,24,3))),'-o',1:144,squeeze(cumsum(QScore10(:,i+9,24,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,24,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,24,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,24,3))),'-o',1:144,squeeze(cumsum(QScore90(:,i+9,24,4))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([])%[1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_0_h=24.jpg")

%% Save results to mat file
save(sprintf('%s.mat','FORECASTING_0'),'-mat');
