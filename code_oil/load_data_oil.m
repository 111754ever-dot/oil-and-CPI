function [x,xlag,T,n,k,g,ng,dates,namesX,namesYE,tcode] = load_data_oil(inflindx,outpindx,standar,r,p,interF,quant,dfm)
%% load_data_oil.m
%  为本文(油价->多国通胀 QFAVAR)定制的数据读取函数。
%  读取 all_data.xlsx 的 EA(国别) 与 Global(全球) 两个表, 返回主程序所需的全部量。
%
%  与作者 load_data.m 的关系:
%   - 返回值签名完全相同: [x,xlag,T,n,k,g,ng,dates,namesX,namesYE,tcode]
%     所以主程序只需把调用 load_data(...) 改成 load_data_oil(...) 即可。
%   - 因为我们的数据已在 Python 端处理为平稳(同比/增长率)并预填缺失,
%     这里【不做】任何平稳化变换(transxFAVAR), tcode 全设为 1(无变换)。
%   - 输入里的 inflindx, outpindx 仅为兼容作者签名而保留, 本函数不使用
%     (我们只有一种通胀口径、一种产出口径)。
%
%  依赖: 当前(或 data/)目录下存在 all_data.xlsx (你已上传到 main)。
% ------------------------------------------------------------------------
nq = numel(quant);                 % 分位数个数

%% ==== 1) 读取国别数据 (工作表 EA) ====
%  'VariableNamingRule','preserve' 保留原列名(CPI_BR 等), 不被 MATLAB 改写。
TEA   = readtable('all_data.xlsx','Sheet','EA','VariableNamingRule','preserve');
xnm   = TEA.Properties.VariableNames(2:end);   % 1x64 cell: {'CPI_BR','CPI_CA',...,'NEER_US'}
x     = TEA{:,2:end};                           % T x 64 数值矩阵(国别变量)
datecol = TEA{:,1};                             % 第一列: 日期

%% ==== 2) 读取全球数据 (工作表 Global) ====
TGL     = readtable('all_data.xlsx','Sheet','Global','VariableNamingRule','preserve');
namesYE = TGL.Properties.VariableNames(2:end);  % 1x5 cell: {'oil_prod','IGREA','loil_price','GSCPI','GEPU'}
g       = TGL{:,2:end};                          % T x 5 数值矩阵(全球变量)

%% ==== 3) 日期 -> 主程序需要的 'yyyyMMM' 字符串(如 '2005Jul') ====
%  主程序里用 datetime(dates,'InputFormat','yyyyMMM') 解析。
if ~isdatetime(datecol); datecol = datetime(datecol); end
dates = cellstr(string(datecol,'yyyyMMM'));      % T x 1 cell

%% ==== 4) 变量名格式化: 'CPI_BR' -> 'CPI.BR' ====
%  主程序用 extractBefore(names,'.') 取"变量名"(CPI/IP/RATE/NEER), 需要以 '.' 分隔。
namesX = strrep(xnm,'_','.')';                   % T? 不, 是 n x 1 cell(列向量)

%% ==== 5) 不做平稳化变换: tcode 全设 1(无变换) ====
[T,n] = size(x);                                 % T=248(月), n=64(国别序列数)
tcode = ones(1,n);

%% ==== 6) xlag: x 的一阶滞后(仅当 AR1x=1 时用到; 基准 AR1x=0 不使用) ====
%  为了不损失观测, 这里不丢首行, 首行用自身占位(内容在 AR1x=0 时无影响)。
xlag = [x(1,:); x(1:end-1,:)];

%% ==== 7) 若估计 QDFM(dfm=1): 丢弃全球变量 g ====
if dfm == 1
    g = [];
end
ng = size(g,2);                                  % 全球变量数(QFAVAR: 5; QDFM: 0)

%% ==== 8) 标准化(完全沿用作者逻辑: standar=1 只标准化 x; =2 连 g 一起) ====
if standar > 0
    x = zscore(x);
    if standar == 2 && ~isempty(g)
        g = zscore(g);
    end
end

%% ==== 9) VAR 每个方程的参数个数 ====
%  k = (r*nq + ng)*p + interF ; 本文 = (4*3 + 5)*2 + 0 = 34
k = (r*nq + ng)*p + interF;

end
