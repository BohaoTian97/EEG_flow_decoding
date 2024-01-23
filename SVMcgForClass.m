%
% author: Bohao Tian, Shijun Zhang, Beihang University
% for EEG flow decoding (by coherence features)
% update on 23rd, Jan, 2023

function [acc,cm,auc] = SVMcgForClass(label_set,data_set,cmin,cmax,gmin,gmax,cstep,gstep,v)
%% X:c Y:g cg:CVaccuracy
%  [cmin,cmax,gmin,gmax,cstep,gstep,v] = deal(-8,8,-8,8,0.1,0.1,10);
%  data_set=squeeze(data_set(:,:,3,1));
%  label_set=squeeze(label_set(:,1));
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
n_count = 8;
acc = zeros(m,n,n_count);
cm = zeros(m,n,n_count,2,2);
auc = zeros(m,n,n_count);
eps = 1e-3;
%% record acc with different c & g,and find the bestacc with the smallest c
basenum = 2;
indices = crossvalind('Kfold', label_set, v);
for k = 1:v
    test = (indices == k); train_cv = ~test;
    X_train = squeeze(data_set(train_cv, :));
    X_test = squeeze(data_set(test, :));
    y_train = label_set(train_cv);
    y_test = label_set(test);
        for i = 1:m
            for j = 1:n
                cmd1 = ['-s 0 -t 2',' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -m ',num2str(4096),' -e ', num2str(eps),' -q'];
                cmd2 = '-q';
                model = libsvmtrain(y_train, X_train, cmd1);
                [predicted_label, a, decision_values] = libsvmpredict(y_test, X_test, model, cmd2);
                [cm_temp, ~] = confusionmat(y_test, predicted_label, 'ORDER', [1,0]);
                [~, ~, ~, auc_temp] = perfcurve(y_test, decision_values, 1);
                acc(i,j) = acc(i,j) + a(1,1);
                cm(i,j,:,:) = squeeze(cm(i,j,:,:)) + cm_temp;
                auc(i,j) = auc(i,j) + auc_temp;
            end
        end
end
    disp(strcat('Processing:', num2str(k), '/', num2str(v)));
%end
acc = acc / v;
cm = cm / v;
auc = auc / v;