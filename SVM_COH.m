%
% author: Bohao Tian, Shijun Zhang, Beihang University
% for EEG flow decoding (by coherence features)
% update on 23rd, Jan, 2023

load('COH_init_data.mat');%load data
disp('Data loaded');

%% organize data and PCA
data_setline=zeros(180,1770,4,subjects);
for i=1:180
    for j=1:4
        for k=1:subjects
            for m=1:60
                data_setnew(i,m,1:m,j,k)=0;
            end
        end
    end
end
for i=1:180
    for j=1:4
        for k=1:subjects
            aline=reshape(data_setnew(i,:,:,j,k),[1,3600]);
            aline(:,all(aline==0,1))= [];
            data_setline(i,:,j,k)=aline;
            clear aline;
        end
    end
end
% reshape a line from column2 to column 59 along each row    
% 180*1770*4*31 data_setline
reduced_data_set=zeros(180,179,4,31);
for i=1:subjects
    for j=1:4
        [~, score, ~] = pca(squeeze(data_setline(:, :, j, i)));
        reduced_data_set(:,:,j,i)=score; %180*179*4*31
    end
end
%% define parameters
[cmin,cmax,gmin,gmax,cstep,gstep,v] = deal(-8,8,-8,8,0.1,0.1,10);
subjects=30;
bestacc = zeros(subjects, 1);
bestc = zeros(subjects, 1);
bestg = zeros(subjects, 1);
best_cm = zeros(subjects, 2, 2);
best_auc = zeros(subjects, 1);
n_shape = zeros(subjects, 3);

[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
n_count = 4;

acc = zeros(subjects,m,n,n_count);
cm = zeros(subjects,m,n,n_count,2,2);
auc = zeros(subjects,m,n,n_count);
%% decoding
% Parallel computation
if isempty(gcp('nocreate'))
   parpool(min([8, subjects]));
end
for N=1:n_count 
    parfor subj = 1:subjects
      [acc(subj,:,N), cm(subj,:,:,:,N), auc(subj,:,N)] = SVMcgForClass(label_set(:, subj), squeeze(reduced_data_set(:, 1:40*N,3, subj)), cmin,cmax,gmin,gmax,cstep,gstep,v);
       % data set 180*1770
       disp(strcat('subject ', num2str(subj), ' done.'));
     end
end
%% get best parameters
load('acc.mat');
load('auc.mat');
load('cm.mat');

for k = 1:n_count
    for subj = 1:subjects
        for i = 1:m
            for j = 1:n
                if acc(subj,i,j,k) > bestacc(subj)
                    bestacc(subj) = acc(subj,i,j,k);
                    bestc(subj) = 2^X(i,j);
                    bestg(subj) = 2^Y(i,j);
                    best_cm(subj,:,:) = cm(subj,i,j,k,:,:);
                    best_auc(subj) = auc(subj,i,j,k);
                elseif (acc(subj,i,j,k) == bestacc(subj)) && (2^X(i,j) < bestc(subj))
                    bestacc(subj) = acc(subj,i,j,k);
                    bestc(subj) = 2^X(i,j);
                    bestg(subj) = 2^Y(i,j);
                    best_cm(subj,:,:) = cm(subj,i,j,k,:,:);
                    best_auc(subj) = auc(subj,i,j,k);
                end
            end
        end
    end
    tag = strcat('thetaCOH_PCA', num2str(40*k));
    save(strcat('classify_coh\', tag, '.mat'), 'bestacc', 'bestc', 'bestg', 'best_cm', 'best_auc', '-v7.3');
end
   
