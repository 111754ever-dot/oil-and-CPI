% Forecasting using Bayesian QFAVAR
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% First version: 07 October 2020
% This version:  November 2021
%==========================================================================

% Reset everything, clear memory and screen
clear; close all; clc;
% Start clock
tic;

% Add path of random number generators
addpath('functions')
addpath('data')
 
%===========================| USER INPUT |=================================
% Model specification
quant = [10,50,90]./100;     % Specify the quantiles to estimate
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
var_sv    = 1;           % VAR variances, 0: constant; 1: time-varying (stochastic volatility) 

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
CQs      = zeros(anumber,nq,varnum,h,6);  % Save raw conditional quantiles   
QScore10 = zeros(anumber,varnum,h,6);     % Quantile score 10%
QScore90 = zeros(anumber,varnum,h,6);     % Quantile score 90%
PL       = zeros(anumber,varnum,h,6);     % Predictive likelihood (based on kernel smoothing)
PITS     = zeros(anumber,varnum,h,6);     % PITS (based on kernel smoothing)
PE       = zeros(anumber,varnum,h,6);     % Prediction Error   
MSPE     = zeros(anumber,varnum,h,6);     % Mean square prediction error
    
for sample = T_thres:T_full
    clc;
    disp('  ')           
    fprintf(['<strong>YOU ARE RUNNING SAMPLE ' num2str(sample) ' OF ' num2str(T_full) '</strong> \n'] )    
    disp('  ')        
    xf_true = x(sample+1:sample+h,:);
        
    %% QFAVAR-SV forecasts
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

    %% FAVAR-SV forecasts
    [xforeMR] = FAVAR_FORE2(x(1:sample,:),xlag(1:sample,:),sample,n,(r+ng)*p+interF,g(1:sample,:),ng,r,p,h,interX,interF,AR1x,incldg,var_sv,nsave,nburn,nthin);
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants                         = quantile(xforeMR(ivar,nfore,:),quant);   
            CQs(sample-T_thres+1,:,ivar,nfore,2)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,2)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,2)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,2)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,2)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,2)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,2)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end

    %% Univariate UCSV models
    for ivar = 1:18
        for nfore = 1:h
            [xforeMR] = TVPSVMR(x(1+nfore:sample,ivar),[ones(sample-nfore,1)],[1],nsave,nburn,1,1,3);
            post_mean_quants                         = quantile(xforeMR,quant); 
            CQs(sample-T_thres+1,:,ivar,nfore,3)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,3)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,3)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,3)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,3)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,3)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,3)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;            
        end
    end

    %% Univariate AR(2)-SV models
    for ivar = 1:18
        for nfore = 1:h
            [xforeMR] = TVPSVMR(x(2+nfore:sample,ivar),[ones(sample-nfore-1,1) x(2:sample-nfore,ivar) x(1:sample-nfore-1,ivar)],[1 x(sample,ivar) x(sample-1,ivar)],nsave,nburn,0,1,3);
            post_mean_quants                         = quantile(xforeMR,quant); 
            CQs(sample-T_thres+1,:,ivar,nfore,4)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,4)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,4)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,4)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,4)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,4)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,4)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;            
        end
    end


    %% Country-specific VAR-SV models
    temp = cell(9,1);
    for icountry = 1:9
        [temp{icountry,1}] = BVAR_FORE(x(1:sample,icountry:9:varnum),sample,r,(r+ng)*p+1,g(1:sample,:),ng,p,h,1,1,nsave,nburn,nthin);
    end
    xforeMR = zeros(n,h,nsave/nthin);
    for i = 1:5
        for j = 1:9
            xforeMR((i-1)*9+j,:,:) = temp{j,1}(i,:,:);
        end
    end
    % Create forecast statistics
    for ivar = 1:varnum
        for nfore = 1:h
            post_mean_quants                         = quantile(xforeMR(ivar,nfore,:),quant);   
            CQs(sample-T_thres+1,:,ivar,nfore,6)     = sort(post_mean_quants);
            QScore10(sample-T_thres+1,ivar,nfore,6)  = (xf_true(nfore,ivar) - post_mean_quants(q10))*(double(xf_true(nfore,ivar)<=post_mean_quants(q10)) - quant(q10));
            QScore90(sample-T_thres+1,ivar,nfore,6)  = (xf_true(nfore,ivar) - post_mean_quants(q90))*(double(xf_true(nfore,ivar)<=post_mean_quants(q90)) - quant(q90));       
            PL(sample-T_thres+1,ivar,nfore,6)        = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Bandwidth',0.7);
            PITS(sample-T_thres+1,ivar,nfore,6)      = ksdensity(post_mean_quants,xf_true(nfore,ivar),'Function','cdf');
            PE(sample-T_thres+1,ivar,nfore,6)        = (xf_true(nfore,ivar)-post_mean_quants(qmed));        
            MSPE(sample-T_thres+1,ivar,nfore,6)      = (xf_true(nfore,ivar)-post_mean_quants(qmed))^2;
        end
    end    

end %recursive forecasts


pdates = dates(T_thres+1:T_full+1);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,1,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,1,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,1,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i,1,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i,1,5))),'--',1:144,squeeze(cumsum(QScore10(:,i,1,6))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,1,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,1,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,1,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i,1,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i,1,5))),'--',1:144,squeeze(cumsum(QScore90(:,i,1,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,1,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,1,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,1,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i+9,1,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i+9,1,5))),'--',1:144,squeeze(cumsum(QScore10(:,i+9,1,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,1,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,1,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,1,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i+9,1,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i+9,1,5))),'--',1:144,squeeze(cumsum(QScore90(:,i+9,1,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_2_h=1.jpg")

pdates = dates(T_thres+6:T_full+6);
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
            xticks([1, 40, 80, 120])
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
            xticks([1, 40, 80, 120])
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
            xticks([1, 40, 80, 120])
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
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
pdates = dates(T_thres+6:T_full+6);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,6,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,6,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,6,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i,6,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i,6,5))),'--',1:144,squeeze(cumsum(QScore10(:,i,6,6))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,6,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,6,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,6,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i,6,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i,6,5))),'--',1:144,squeeze(cumsum(QScore90(:,i,6,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,6,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,6,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,6,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i+9,6,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i+9,6,5))),'--',1:144,squeeze(cumsum(QScore10(:,i+9,6,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,6,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,6,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,6,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i+9,6,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i+9,6,5))),'--',1:144,squeeze(cumsum(QScore90(:,i+9,6,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_2_h=6.jpg")


pdates = dates(T_thres+12:T_full+12);
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,12,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,12,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,12,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i,12,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i,12,5))),'--',1:144,squeeze(cumsum(QScore10(:,i,12,6))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,12,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,12,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,12,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i,12,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i,12,5))),'--',1:144,squeeze(cumsum(QScore90(:,i,12,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,12,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,12,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,12,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i+9,12,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i+9,12,5))),'--',1:144,squeeze(cumsum(QScore10(:,i+9,12,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,12,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,12,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,12,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i+9,12,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i+9,12,5))),'--',1:144,squeeze(cumsum(QScore90(:,i+9,12,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end
saveas(gcf,"forecasting_2_h=12.jpg")


pdates = dates(T_thres+24:T_full+24);
%Plot QScore10, h=1
fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j = 1:4
    for i=1:9
        subplot(4,9,(j-1)*9+i)
        if j == 1
        plot(1:144,squeeze(cumsum(QScore10(:,i,24,1))),'-',1:144,squeeze(cumsum(QScore10(:,i,24,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i,24,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i,24,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i,24,5))),'--',1:144,squeeze(cumsum(QScore10(:,i,24,6))),'-.','LineWidth',2);
                if i == 1
                    ylabel('10\%, HICP');
                end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                 
        elseif j == 2        
            plot(1:144,squeeze(cumsum(QScore90(:,i,24,1))),'-',1:144,squeeze(cumsum(QScore90(:,i,24,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i,24,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i,24,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i,24,5))),'--',1:144,squeeze(cumsum(QScore90(:,i,24,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, HICP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i))                
        elseif j == 3
            plot(1:144,squeeze(cumsum(QScore10(:,i+9,24,1))),'-',1:144,squeeze(cumsum(QScore10(:,i+9,24,2))),':',...
                1:144,squeeze(cumsum(QScore10(:,i+9,24,3))),'--.',1:144,squeeze(cumsum(QScore10(:,i+9,24,4))),'-.',...
                1:144,squeeze(cumsum(QScore10(:,i+9,24,5))),'--',1:144,squeeze(cumsum(QScore10(:,i+9,24,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('10\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        elseif j == 4        
            plot(1:144,squeeze(cumsum(QScore90(:,i+9,24,1))),'-',1:144,squeeze(cumsum(QScore90(:,i+9,24,2))),':',...
                1:144,squeeze(cumsum(QScore90(:,i+9,24,3))),'--.',1:144,squeeze(cumsum(QScore90(:,i+9,24,4))),'-.',...
                1:144,squeeze(cumsum(QScore90(:,i+9,24,5))),'--',1:144,squeeze(cumsum(QScore90(:,i+9,24,6))),'-.','LineWidth',2);
               if i == 1
                    ylabel('90\%, IP');
               end
            grid on;   
            xlim([1 size(QScore10,1)]);
            xticks([1, 40, 80, 120])
            set(gca,'xticklabels',pdates(xticks))
            xtickangle(45)
            box on
            set(gca,'FontSize',12)  
            title(names(i+9))                
        end
    end
end

saveas(gcf,"forecasting_2_h=24.jpg")


%% Save results to mat file
save(sprintf('%s.mat','FORECASTING_2_new'),'-mat');
