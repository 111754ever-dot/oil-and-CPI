function [beta_mat,B,SIGMA,F,L,R,iR,psi,tau] = drawBVARsign(y,x,F,L,R,iR,psi,tau,lam_prmean,lam_iprvar,sg,index_vec,N,p,interF)


est_alg = double(size(x,1)>size(x,2)) + 1;
[T,M] = size(y);
KK = size(x,2);
   
y_til = y - F*L';    
% STEP 1: update BETA
beta_mat = zeros(KK,M);
for ieq = 1:M
    [beta_i,psi(:,ieq),tau(ieq)] = horseshoe(y_til(:,ieq),x,psi(:,ieq),tau(ieq),R(ieq,ieq),est_alg,ieq);      
    beta_mat(:,ieq) = beta_i;
end
B = [beta_mat(interF+1:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];
rej_count = 0;
while max(abs(eig(B))) > 0.999
    rej_count = rej_count + 1;   
    if rej_count > 500; y_til  = y; end
    for ieq = 1:M           
        [beta_i,psi(:,ieq),tau(ieq)] = horseshoe(y_til(:,ieq),x,psi(:,ieq),tau(ieq),R(ieq,ieq),est_alg,ieq);
        beta_mat(:,ieq) = beta_i;         
    end      
    B = [beta_mat(interF+1:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];
end
    
% STEP 2: update SIGMA 
yhat = y - x*beta_mat;  % Now yhat is the VAR residual that follows the factor model          
    
% Sample shocks
for t = 1:T
    F_var  = inv(1*eye(N) + L'*diag(iR)*L);
    F_mean = F_var*L'*diag(iR)*yhat(t,:)';
    F(t,:) = (F_mean + chol(F_var)'*randn(N,1))';
end
F = F - mean(F); % demean factor sample
    
% I am doing now a very inefficient element-by-element update of the posterior of lambda,
% so I can make a straightfoward use of the univariate truncated normal generator 
for i = 1:M        
    Lvar = diag(lam_iprvar(i,:)) + (F'*F).*iR(i); % inverse of posterior covariance matrix of L(i,:)       
    L_bar(i,:) = Lvar\(diag(lam_iprvar(i,:))*lam_prmean(i,:)' + (F'*yhat(:,i)).*iR(i));
    for j = 1:N
        whole_vec = Lvar(j,:)'.*(L(i,:) - L_bar(i,:));
        Lpostvar = 1./Lvar(j,j); ss = sqrt(Lpostvar);
        Lpostmean = L_bar(i,j) - sum(whole_vec(find(index_vec~=j)))*Lpostvar;            
        if sg(i,j)  == 1
            lower  = (0 - Lpostmean)/ss;   
            upper  = (10000 - Lpostmean)/ss;
            L(i,j) = trandn(lower,upper);
            L(i,j) = Lpostmean + ss*L(i,j);           
        elseif sg(i,j)  == -1
            lower  = (-10000 - Lpostmean)/ss;
            upper  = (0 - Lpostmean)/ss;
            L(i,j) = trandn(lower,upper);
            L(i,j) = Lpostmean + ss*L(i,j);
        elseif sg(i,j) == 0
            L(i,j)  = 0;          
        elseif isnan(sg(i,j))
            L(i,j)  = Lpostmean + ss*randn;
        end
    end
end
    
% draw R
sse2    = sum((yhat - F*L').^2);
R_1     = (1 + T);    
R_2     = (0.01 + sse2);
iR      = gamrnd(R_1./2,2./R_2);
R       = diag(1./iR);
    
SIGMA  = L*cov(F)*L' + R;