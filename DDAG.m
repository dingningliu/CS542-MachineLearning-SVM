function predict_DDAG = DDAG(item)

%item is one row of scores_ovo matrix
label = 0:9;
while length(label) > 1
    %eg:classify the first layer 1 or 9, then minimize the size of label
    if item( label(1)*9+label(end) ) < 0
        label = label(2:end);
    else
        label = label(1:end-1);
    end
end
predict_DDAG = label;
end