load('/Users/liudingning/Desktop/sum/ml542/hw4/ps4_2019_files/MNIST.mat');
%display the i-th image, the shape of digit;
%imagesc(reshape(train_samples(i,:),[28,28]));

%%
% SVM classifier with One versus the rest
% Solve a, b, kernel, get the score for the testing data
a = zeros(4000,10);   % We have 10 svms
b = zeros(10,1);
scores_ovr = zeros(1000,10);
C = 10;
poly = 6;


for i = 1:10
    [a(:,i), b(i), scores_ovr(:,i)] = svm_ovr(i-1, C, train_samples_labels, train_samples, test_samples, poly);
end

[~, loc] = max(scores_ovr, [] , 2);
loc = loc - 1;
right_ova = sum(loc == test_samples_labels);    %the number of accurate lable
accurate_ova = right_ova/length(loc);   %the accuracy rate
disp(confusionmat(test_samples_labels, loc));


%% 
% SVM classifier with One versus One
% Solve a, b, kernel, get the score for the testing data
scores_ovo = zeros(1000,90);  % we have 10*(10-1)=90 SVMs
C = 10;
k = 0; 


for i = 0:9
    for j = 0:9
        if j ~= i
            k = k + 1 ;
            scores_ovo(:,k) = svm_ovo(i,j, C, train_samples_labels, train_samples, test_samples, 2);
        end
    end
end


%the score matrix has 90 cols, turn it into 10 cols matrix for each label
%sign function can return 1/0/-1, determine the label
%is a column vector containing the sum of each row.
predict_label_ovo = zeros(1000,10);
for i = 1:10
    predict_label_ovo(:,i) =  sum(sign(scores_ovo(:, (i-1)*9+1:i*9)),2);
end



[~, loc_ovo] = max(predict_label_ovo, [] , 2);
loc_ovo = loc_ovo - 1;
right_ovo = sum(loc_ovo == test_samples_labels);    %the number of accurate lable
accurate_ovo = right_ovo/length(loc_ovo);   %the accuracy rate
disp(confusionmat(test_samples_labels, loc_ovo));


%%

predict_label_dag = zeros(1000,1);

for i = 1:1000
     predict_label_dag(i) = DDAG(scores_ovo(i,:)); %get the predicted label from DDAG
end

    

right_dag = sum(predict_label_dag == test_samples_labels);
accurate_dag = right_dag/length(predict_label_dag);   %the accuracy rate
disp(confusionmat(test_samples_labels, predict_label_dag));







