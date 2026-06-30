function [beta] = FFBSVB(y,H,F0,F,V,W,r)

[T,n] = size(y);
p = size(F,2);
beta = zeros(p,T);

a = zeros(p,T);
m = zeros(p,T);
R = zeros(p,p,T);
C = repmat(4*eye(p),1,1,T);
f = zeros(n,T);
q = zeros(n,n,T);
e = zeros(n,T);
K = zeros(p,n,T);

% Forward filtering
for t = 1:T
    % Update time t prior
    a(:,t) = F0(t,:)' + F*m(:,max(1,t-1));
    R(:,:,t) = F*C(:,:,max(1,t-1))*F' + W(:,:,t);
    % One step ahead predictive distributions
    f(:,t) = H(:,:,t)*a(1:r,t);
    q(:,:,t) = H(:,:,t)*R(1:r,1:r,t)*H(:,:,t)' + diag(V);
    e(:,t) = y(t,:)' - f(:,t);
    % Time t posterior
    K(:,:,t) = (R(:,1:r,t)*H(:,:,t)')/q(:,:,t);
    m(:,t) = a(:,t) + K(:,:,t)*e(:,t);
    C(:,:,t) = (R(:,:,t) - K(:,:,t)*H(:,:,t)*R(1:r,:,t));
end
%beta = m(1:r,:)';

B = zeros(p,r,T);
aT = zeros(p,T);    aT(:,T) = m(:,T);
RT = zeros(p,p,T);  RT(:,:,T) = C(:,:,T);

% Backwards sampling
beta(:,T) = aT(:,T);
for t=T-1:-1:1
    B(:,:,t) = (C(:,:,t)*F(1:r,:)')/R(1:r,1:r,t+1);
    aT(:,t) = m(:,t) + B(:,:,t)*(beta(1:r,t+1) + F0(t+1,1:r)' - a(1:r,t+1));
    RT(:,:,t) = (C(:,:,t) - B(:,:,t)*F(1:r,:)*C(:,:,t));

    beta(:,t) = aT(:,t);
end

beta = beta(1:r,:)';
end