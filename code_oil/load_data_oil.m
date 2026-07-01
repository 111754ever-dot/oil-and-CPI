function [x,xlag,T,n,k,g,ng,dates,namesX,namesYE,tcode] = load_data_oil(inflindx,outpindx,standar,r,p,interF,quant,dfm)
%% load_data_oil.m
%  为本文(油价->多国通胀 QFAVAR)定制的数据读取函数。
%  读取 all_data.xlsx 的 EA(国别) 与 Global(全球) 两个表, 返回主程序所需的全部量。
%
%  与作者 load_data.m 的关系:
%   - 返回值签名完全相同: [x,xlag,T,n,k,g,ng,dates,namesX,namesYE,tcode]
%     主程序只需把 load_data(...) 改成 load_data_oil(...)。
%   - 我们的数据已在 Python 端处理为平稳(同比/增长率)并预填缺失, 故【不做】
%     平稳化变换(transxFAVAR), tcode 全设 1(无变换)。
%   - inflindx, outpindx 仅为兼容作者签名保留, 本函数不使用。
%
%  【修订记录】
%   (1) 硬化"文本型数字": 用 tbl2num 把可能被存成文本的数字列稳妥转 double,
%       避免 readtable 把整列读成字符串导致后续矩阵运算/ zscore 失败。
%       (建议直接使用本目录下已清洗为纯数值的 all_data.xlsx; 用原文件也可,
%        tbl2num 会兜底。)
%   (2) xlag: 保留全部 248 个月(基准 AR1x=0 不使用 xlag, 保留数据最优)。
%       首期用自身占位, 在 AR1x=0 下无任何影响。若将来设定 AR1x=1, 请见
%       下方第 6 步的注释, 取消两行注释以丢弃首期得到正确滞后。
% ------------------------------------------------------------------------
nq = numel(quant);

%% ==== 1) 读取国别数据 (工作表 EA), 并稳妥转数值 ====
TEA   = readtable('all_data.xlsx','Sheet','EA','VariableNamingRule','preserve');
xnm   = TEA.Properties.VariableNames(2:end);   % {'CPI_BR',...,'NEER_US'}
x     = tbl2num(TEA(:,2:end));                  % T0 x 64 数值(硬化文本型数字)
datecol = TEA{:,1};                             % 日期列

%% ==== 2) 读取全球数据 (工作表 Global), 并稳妥转数值 ====
TGL     = readtable('all_data.xlsx','Sheet','Global','VariableNamingRule','preserve');
namesYE = TGL.Properties.VariableNames(2:end);  % {'oil_prod','IGREA','loil_price','GSCPI','GEPU'}
g       = tbl2num(TGL(:,2:end));                 % T0 x 5 数值

%% ==== 3) xlag: 一阶滞后(基准 AR1x=0 不使用, 保留全部 248 个月) ====
xlag = [x(1,:); x(1:end-1,:)];   % 首期用自身占位(AR1x=0 下无影响)
% —— 若将来设定 AR1x=1, 取消下面两行注释以得到"真实滞后+丢首期"(样本 248->247):
% xlag = x(1:end-1,:);  x = x(2:end,:);  g = g(2:end,:);  datecol = datecol(2:end);

%% ==== 4) 日期 -> 'yyyyMMM' 字符串(如 '2005Jul') ====
if ~isdatetime(datecol); datecol = datetime(datecol); end
dates = cellstr(string(datecol,'yyyyMMM'));

%% ==== 5) 变量名 'CPI_BR' -> 'CPI.BR' (主程序用 extractBefore(names,'.')) ====
namesX = strrep(xnm,'_','.')';                   % n x 1 cell

%% ==== 6) 不做平稳化: tcode 全设 1 ====
[T,n] = size(x);                                 % T=248, n=64
tcode = ones(1,n);

%% ==== 7) QDFM(dfm=1): 丢弃全球变量 ====
if dfm == 1
    g = [];
end
ng = size(g,2);

%% ==== 8) 标准化(沿用作者逻辑) ====
if standar > 0
    x = zscore(x);
    if standar == 2 && ~isempty(g)
        g = zscore(g);
    end
end

%% ==== 9) VAR 每方程参数个数: k = (r*nq+ng)*p + interF = (4*3+5)*2 = 34 ====
k = (r*nq + ng)*p + interF;

end

% ========================================================================
function M = tbl2num(tb)
% 把 table 每一列稳妥转成 double, 处理"文本型数字"(Excel 把数字存成了文本)。
M = zeros(height(tb), width(tb));
for j = 1:width(tb)
    col = tb{:,j};
    if isnumeric(col)
        M(:,j) = col;                 % 已是数值, 直接用
    else
        M(:,j) = str2double(string(col));   % 文本型数字 -> double
    end
end
if any(isnan(M(:)))
    warning('load_data_oil:NaNafterConvert', ...
        '转换后存在 NaN, 请检查 all_data.xlsx 是否有非数值/空单元格。');
end
end
