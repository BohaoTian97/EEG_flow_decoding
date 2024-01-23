%
% author: Bohao Tian, Shijun Zhang, Beihang University
% for EEG flow decoding (by coherence features)
% update on 23rd, Jan, 2023

load('PSD_init_data.mat');%load data
disp('Data loaded');

%% define parameters
[cmin,cmax,gmin,gmax,cstep,gstep,v] = deal(-8,8,-8,8,0.1,0.1,10);
bestacc = zeros(subjects, 1);
bestc = zeros(subjects, 1);
bestg = zeros(subjects, 1);
best_cm = zeros(subjects, 2, 2);
best_auc = zeros(subjects, 1);
n_shape = zeros(subjects, 3);

[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);

 acc = zeros(subjects,m,n);
 cm = zeros(subjects,m,n,2,2);
 auc = zeros(subjects,m,n);
%% decoding
% Parallel computation
if isempty(gcp('nocreate'))
   parpool(min([8, subjects]));
end
parfor subj = 1:subjects
    [acc(subj,:,:), cm(subj,:,:,:,:), auc(subj,:,:)] = SVMcgForClass(label_set(:, subj), squeeze(data_set(:, :,3, subj)), cmin,cmax,gmin,gmax,cstep,gstep,v);
    disp(strcat('subject ', num2str(subj), ' done.'));
end

for subj = 1:subjects
        for i = 1:m
            for j = 1:n
                if acc(subj,i,j) > bestacc(subj)
                    bestacc(subj) = acc(subj,i,j);
                    bestc(subj) = 2^X(i,j);
                    bestg(subj) = 2^Y(i,j);
                    best_cm(subj,:,:) = cm(subj,i,j,:,:);
                    best_auc(subj) = auc(subj,i,j);
                elseif (acc(subj,i,j) == bestacc(subj)) && (2^X(i,j) < bestc(subj))
                    bestacc(subj) = acc(subj,i,j);
                    bestc(subj) = 2^X(i,j);
                    bestg(subj) = 2^Y(i,j);
                    best_cm(subj,:,:) = cm(subj,i,j,:,:);
                    best_auc(subj) = auc(subj,i,j);
                end
            end
        end
end
save('classify_psd\alpha\svmpsd_alpha.mat', 'bestacc', 'bestc', 'bestg', 'best_cm', 'best_auc', '-v7.3');
