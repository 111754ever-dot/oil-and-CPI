function [x,xlag,T,n,k,g,ng,dates,namesX,namesYE,tcode] = load_data(inflindx,outpindx,standar,r,p,interF,quant,dfm)
%% Function to euroarea and global data for QFAVAR model
if isempty(quant)
    nq=1;
else
    nq = length(quant);
end

%% ======|Load series to extract quantile factors from
warning off;
[data,text]=xlsread('all_data.xlsx','EA');
x     = data(2:end,:);
tcode = data(1,:);
dates = text(6:end,1);
names = text(1,2:end)';

% Choose inflation measure here
x  = x(:,[(inflindx-1)*9+(1:9),28:72]);
namesX = names([(inflindx-1)*9+(1:9),28:72]);
tcode = tcode([(inflindx-1)*9+(1:9),28:72]);

% Default order is INFL, UNEMP, LTIR, ESI, CLIFS, NPCR; decide here whether to switch from UNEMP to IP
if outpindx ~= 1
    x(:,10:18) = data(2:end,73:81);
    tcode(10:18) = 7;
    namesX(10:18) = names(73:81);
end

% Update July 2022: Remove NPCR series from dataset
x = x(:,1:45);
namesX = namesX(1:45);
tcode = tcode(1:45);

% Transform to stationarity
[x,xlag,dates] = transxFAVAR(x,dates,tcode);

%% ======|Load observed (global) factors
[g,B]=xlsread('all_data.xlsx','Global');
namesYE = B(1,2:end);
% Global variables are already transformed, adjust for stationarity tranformations in x
if sum(tcode==7)>0       
    g = g(13:end,:);
else
    g = g(2:end,:);
end
% note that x has one less obs because of taking first lag (whether this lag is used in the model or not)
g = g(2:end,:);
g(:,1) = adjout(g(:,1),4,5);

% Check whether data are desired to be standardized
if standar > 0
    x = zscore(x);
    if standar == 2
        g = zscore(g);
    end
end

% The DFM is simply the favar with no g, i.e. only x variables are used to extract (quantile) factors
if dfm == 1
    g = [];
end

% Obtain number of series in x, and number of VAR parameters based on r, p and the presence of an intercept (interF).
[T,n] = size(x);           % dimensions of x
ng    = size(g,2);         % number of additional predictors in measurement equation
k     = (r*nq + ng)*p + interF;     % number of var parameters per equation
warning on;