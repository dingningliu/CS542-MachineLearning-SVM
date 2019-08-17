function [a,b,score] = svm_ovr(label, C, train_label, train, test, poly)
%lable :from 0 to 9
%C :slack variabel
%train_label/train :trainging data y & x
%test : testing data x

%data processing
%seperate data into two labels
ind_y = train_label == label;
ind_n = train_label ~= label;

%let y=1 for label ind_y , y=-1 for ind_n 
t = train_label;   
t(ind_y) = 1;
t(ind_n) = -1;
T = diag(t);   %diagonal matrix(4000x4000)
%kernel
k = polykernel(train,train,poly);  % matrix(4000x4000)

%quadratic programming to solve vector a
% min (1/2)*x'*H*x + f'*x
% by lagagian: max w(a)= a - (1/2)*t_i*t_j*a_i*a_j*k
% that is min -a + (1/2)*t_i*t_j*a_i*a_j*k
% s.t. y*a = 0
H = T * k * T;           % matrix(4000x4000)
f = -1 * ones(4000,1);       % vector(4000x1)
Aeq = t';                    % vector(1x4000)
beq = 0;
lb = zeros(4000,1);
ub = C * ones(4000,1);
a = quadprog(H,f,[],[],Aeq,beq,lb,ub);   
%b = (sum(t) - a.' * diag(t) * polykernel(train, train, poly) * diag(t) * a) / length(sv_ind);

b = (sum(t) - (a' * T * k * T * a)) / length(a);  


% w = a*t*x
% h(x) = g(w'x+b)   w'x+b = a*t*x'*x + b
% score = (a.' * T * (train * test.').^6 + b).'
score = a' * T * polykernel(train, test, 6) + b;

end




