function score = svm_ovo(label_1, label_2, C, train_label, train, test, ~)
%lable_1/_2 :from 0 to 9, for one versus one, we classify between two labels
%C :slack variabel
%train_label/train :trainging data y & x
%test : testing data x
%sigma : parameter of gaussian kernel

%data processing
%select two labels, then we have l*(l-1)/2 SVMs, for one SVM
ind_1 = find(train_label == label_1);
ind_2 = find(train_label == label_2);
train_two = train([ind_1; ind_2], : );    % x

t = train_label;
t(ind_1) = 1;
t(ind_2) = -1;
t_two = t([ind_1;ind_2], : );
T_two = diag(t_two); 

n = length(ind_1) + length(ind_2);

%kernerl gaussian
%k = gaussiankernel(train,train,sigma);
k = polykernel(train_two,train_two,6);

% min (1/2)*x'*H*x + f'*x
% by lagagian: max w(a)= a - (1/2)*t_i*t_j*a_i*a_j*k
% that is min -a + (1/2)*t_i*t_j*a_i*a_j*k
% s.t. y*a = 0
H = T_two * k * T_two;           
f = -1 * ones(n,1);      
Aeq = t_two';                   
beq = 0;
lb = zeros(n,1);
ub = C * ones(n,1);
a = quadprog(H,f,[],[],Aeq,beq,lb,ub);  
%b = (sum(t) - a.' * diag(t) * polykernel(train, train, poly) * diag(t) * a) / length(sv_ind);

b = (sum(t_two) - (a' * T_two * k * T_two * a)) / length(a);  


% w = a*t*x
% h(x) = g(w'x+b)   w'x+b = a*t*x'*x + b
% score = (a.' * T * (train * test.').^6 + b).'
score = a' * T_two * polykernel(train_two, test, 6) + b;

end





