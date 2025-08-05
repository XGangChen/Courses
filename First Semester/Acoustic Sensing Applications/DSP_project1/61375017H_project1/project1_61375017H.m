clc;
clear;
tic;
%Normal/FD
% clear
in_path='./test/input/';   % 輸入語音檔路徑
out_path='./test/output/'; % 輸出語音檔路徑
%cd (in_path);
file=dir(in_path);
for h=3:1:length(file)
    % cd(in_path);
    in_path_p=[in_path,file(h).name];
    indir=in_path_p;
    indir2 = './input'
    out_path_p=[out_path,file(h).name];
    outdir=out_path_p;
    outdir2 = './output'
    out_ext='.mfc';
    out_ext2 = '.mfc'
    mfcc_all_v1(indir,outdir,out_ext);
    mfcc_all_v1(indir2,outdir2,out_ext2);
    fprintf('complete\n');
    fprintf('NORMAL complete\n')
end

toc;

[x1, fs1] = audioread('./input/im/cancer/C96976-2.wav');
[x2, fs2] = audioread('./input/nor/22-2.wav');

time1 = (1:length(x1))/fs1;
time2 = (1:length(x2))/fs2;

subplot(221);
plot(time1, x1);
subplot(222);
plot(time2, x2);
subplot(223);
spectrogram(x1, 1024, 1000, [], fs1, 'yaxis');
subplot(224);
spectrogram(x2, 1024, 1000, [], fs2, 'yaxis');

%%----------------------------------------------------------------------------------------------------------------
% Define the path containing the .mfc files
% Replace 'Select path' with the actual path
path1 = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/output/nor/'; 
path2 = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/output/im/cancer/';
path3 = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/output/im/CYST/';

cd(path1)              % Change directory to the specified path
% Get the list of .mfc files in the folder and sort them by name
fileList1 = dir(fullfile(path1, '*.mfc'));
[~, idx] = sort({fileList1.name});
fileList1 = fileList1(idx);

cd(path2)              % Change directory to the specified path
% Get the list of .mfc files in the folder and sort them by name
fileList2 = dir(fullfile(path2, '*.mfc'));
[~, idx] = sort({fileList2.name});
fileList2 = fileList2(idx);

cd(path3)              % Change directory to the specified path
% Get the list of .mfc files in the folder and sort them by name
fileList3 = dir(fullfile(path3, '*.mfc'));
[~, idx] = sort({fileList3.name});
fileList3 = fileList3(idx);

% Initialize arrays to store normal and sick data separately
normalTrain = [];
normalTest = [];
sickTrain = [];
sickTest = [];

% Separate normal and sick files based on file naming convention or index
% Assume that files from 1 to 20 are normal and 21 onwards are sick (adjust if needed)
normalFiles = fileList1(1:20); % First 20 files as normal
sickFiles = [fileList2(1:10); fileList3(1:10)]; % Remaining files as sick

cd(path1);
% Load normal data
for i = 1:10
    cepData = load(normalFiles(i).name);      % Load each mfc file
    number_person(i,:) = length(cepData(:,1)); % Record the length of each mfc file
    normalTrain = [normalTrain; cepData];       % Concatenate normal data
end
for i = 11:20
    cepData = load(normalFiles(i).name);      % Load each mfc file
    number_person(i,:) = length(cepData(:,1)); % Record the length of each mfc file
    normalTest = [normalTest; cepData];       % Concatenate normal data
end

cd(path2);
% Load sick data
for i = 1:5
    cepData = load(sickFiles(i).name);        % Load each mfc file
    number_person(i + length(normalFiles),:) = length(cepData(:,1));
    sickTrain = [sickTrain; cepData];           % Concatenate sick data
end
for i = 6:10
    cepData = load(sickFiles(i).name);        % Load each mfc file
    number_person(i + length(normalFiles),:) = length(cepData(:,1));
    sickTest = [sickTest; cepData];           % Concatenate sick data
end

cd(path3);
% Load sick data
for i = 11:15
    cepData = load(sickFiles(i).name);        % Load each mfc file
    number_person(i + length(normalFiles),:) = length(cepData(:,1));
    sickTrain = [sickTrain; cepData];           % Concatenate sick data
end
for i = 16:20
    cepData = load(sickFiles(i).name);        % Load each mfc file
    number_person(i + length(normalFiles),:) = length(cepData(:,1));
    sickTest = [sickTest; cepData];           % Concatenate sick data
end

% Combine training and test data in specified order
trainData = [normalTrain; sickTrain];
testData = [normalTest; sickTest];

% Assign labels: 0 for normal, 1 for sick
trainLabels = [zeros(size(normalTrain,1), 1); ones(size(sickTrain,1), 1)];
testLabels = [zeros(size(normalTest,1), 1); ones(size(sickTest,1), 1)];

% Combine labels with data
VoiceTrain = [trainLabels trainData]
VoiceTest = [testLabels testData]

% Define the path for saving .mat files
savePath = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/'; % Replace with the actual path you want to save the files to

cd(savePath);
% Save training and testing data with labels as .mat files at the specified path
save('VoiceTrain.mat', 'VoiceTrain');
save('VoiceTest.mat', 'VoiceTest');

%%------Frame-Accuracy--------------------------------------------------
%-------------K=1---------------------------------------------------------
predict_label1 = trainedModel1.predictFcn(VoiceTest(:, 2:27));

a=0;
for i=1:length(VoiceTest)
    if predict_label1(i,1)==VoiceTest(i,1)
        a=a+1;
    end
end
accuracy_frame1=a/length(VoiceTest)
%-------------K=3---------------------------------------------------------
predict_label3 = trainedModel3.predictFcn(VoiceTest(:, 2:27));
a=0;
for i=1:length(VoiceTest)
    if predict_label3(i,1)==VoiceTest(i,1)
        a=a+1;
    end
end
accuracy_frame3=a/length(VoiceTest)
%-------------K=5---------------------------------------------------------
predict_label5 = trainedModel5.predictFcn(VoiceTest(:, 2:27));
a=0;
for i=1:length(VoiceTest)
    if predict_label5(i,1)==VoiceTest(i,1)
        a=a+1;
    end
end
accuracy_frame5=a/length(VoiceTest)
%-------------K=7---------------------------------------------------------
predict_label7 = trainedModel7.predictFcn(VoiceTest(:, 2:27));
a=0;
for i=1:length(VoiceTest)
    if predict_label7(i,1)==VoiceTest(i,1)
        a=a+1;
    end
end
accuracy_frame7=a/length(VoiceTest)

%% Q2 人的準確度
%% Q2 人的準確度
%%------------K=1----------------------------
label1_1 = zeros(10, 1);

path1 = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/test/nor/';  % 輸入mfc檔路徑資料夾
cd(path1)                                 % 進入選取的資料夾
file1 = dir([path1 '/*.mfc']);

for i = 1:length(file1)
    normal = 0;
    sick = 0;
    cep_data1 = load(file1(i).name, 'ascii');
    [m, n] = size(cep_data1);
    number_person = m;

    for j = 1:number_person
        predict_label1 = trainedModel1.predictFcn(cep_data1(j, :));

        if predict_label1 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if normal/number_person > 0.5
        label1_1(i, 1) = 0;
    else
        label1_1(i, 1) = 1;
    end
end

label2_1 = zeros(10, 1);

path2 = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/test/sick/';  % 輸入mfc檔路徑資料夾
cd(path2)                                 % 進入選取的資料夾
file2 = dir([path2 '/*.mfc']);

for i = 1:length(file2)
    normal = 0;
    sick = 0;
    cep_data2 = load(file2(i).name, 'ascii');
    [m, n] = size(cep_data2);
    number_person = m;

    for j = 1:number_person
        predict_label1 = trainedModel1.predictFcn(cep_data2(j, :));

        if predict_label1 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if sick/number_person > 0.5
        label2_1(i, 1) = 1;
    else
        label2_1(i, 1) = 0;
    end
end

label1 = [label1_1; label2_1];
answer = [zeros(10, 1); ones(10, 1)];
count = 0;

for i = 1:20
    if label1(i, 1) == answer(i, 1)
        count = count + 1;
    end
end

accuracy = count / 20;
disp(['Accuracy for person(K=1): ', num2str(accuracy)]);

%%------------K=3----------------------------
label1_3 = zeros(10, 1);

for i = 1:length(file1)
    normal = 0;
    sick = 0;

    for j = 1:number_person
        predict_label3 = trainedModel3.predictFcn(cep_data1(j, :));

        if predict_label3 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if normal/number_person > 0.5
        label1_3(i, 1) = 0;
    else
        label1_3(i, 1) = 1;
    end
end

label2_3 = zeros(10, 1);

for i = 1:length(file2)
    normal = 0;
    sick = 0;

    for j = 1:number_person
        predict_label3 = trainedModel3.predictFcn(cep_data2(j, :));

        if predict_label3 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if sick/number_person > 0.5
        label2_3(i, 1) = 1;
    else
        label2_3(i, 1) = 0;
    end
end

label3 = [label1_3; label2_3];
answer = [zeros(10, 1); ones(10, 1)];
count = 0;

for i = 1:20
    if label3(i, 1) == answer(i, 1)
        count = count + 1;
    end
end

accuracy = count / 20;
disp(['Accuracy for person(K=3): ', num2str(accuracy)]);

%%------------K=5----------------------------
label1_5 = zeros(10, 1);

for i = 1:length(file1)
    normal = 0;
    sick = 0;

    for j = 1:number_person
        predict_label5 = trainedModel5.predictFcn(cep_data1(j, :));

        if predict_label3 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if normal/number_person > 0.5
        label1_5(i, 1) = 0;
    else
        label1_5(i, 1) = 1;
    end
end

label2_5 = zeros(10, 1);

for i = 1:length(file2)
    normal = 0;
    sick = 0;

    for j = 1:number_person
        predict_label5 = trainedModel5.predictFcn(cep_data2(j, :));

        if predict_label3 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if sick/number_person > 0.5
        label2_5(i, 1) = 1;
    else
        label2_5(i, 1) = 0;
    end
end

label5 = [label1_5; label2_5];
answer = [zeros(10, 1); ones(10, 1)];
count = 0;

for i = 1:20
    if label5(i, 1) == answer(i, 1)
        count = count + 1;
    end
end

accuracy = count / 20;
disp(['Accuracy for person(K=5): ', num2str(accuracy)]);

%%------------K=7----------------------------
label1_7 = zeros(10, 1);

for i = 1:length(file1)
    normal = 0;
    sick = 0;

    for j = 1:number_person
        predict_label7 = trainedModel7.predictFcn(cep_data1(j, :));

        if predict_label7 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if normal/number_person > 0.5
        label1_7(i, 1) = 0;
    else
        label1_7(i, 1) = 1;
    end
end

label2_7 = zeros(10, 1);

for i = 1:length(file2)
    normal = 0;
    sick = 0;

    for j = 1:number_person
        predict_label7 = trainedModel7.predictFcn(cep_data2(j, :));

        if predict_label7 == 0
            normal = normal + 1;
        else
            sick = sick + 1;
        end
    end

    if sick/number_person > 0.5
        label2_7(i, 1) = 1;
    else
        label2_7(i, 1) = 0;
    end
end

label7 = [label1_7; label2_7];
answer = [zeros(10, 1); ones(10, 1)];
count = 0;

for i = 1:20
    if label7(i, 1) == answer(i, 1)
        count = count + 1;
    end
end

accuracy = count / 20;
disp(['Accuracy for person(K=7): ', num2str(accuracy)]);

%%
% 3
[y, fs] = audioread('/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/input/nor/22-2.wav');
sound(y, fs);
time = (1:length(y))/fs;   % 時間軸的向量

subplot(2, 1, 1);
plot(time, y);             % 畫出時間軸上的波形
title('Normal 22-2.wav');
subplot(2, 1, 2);
spectrogram(y, 1024, 1000, [], fs, 'yaxis');

[y, fs] = audioread('/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/input/im/cancer/C96976-2.wav');
sound(y, fs);
time = (1:length(y))/fs;   % 時間軸的向量

figure;
subplot(2, 1, 1);
plot(time, y);             % 畫出時間軸上的波形
title('Cancer C96976-2.wav');
subplot(2, 1, 2);
spectrogram(y, 1024, 1000, [], fs, 'yaxis');

%%
% 4
path = '/home/xgang/XGang/Graduation/First_Year/Acoustic Sensing Applications/DSP_project1/Code/mfcc/output/other/';  % 輸入mfc檔路徑資料夾
cd(path)                                 % 進入選取的資料夾
file=dir([path '/*.mfc'])
DataPre = [];

for i = 1:length(file)
    cep_data = load(file(i).name);            % 讀入每個mfc檔
    number_person(i, 1) = length(cep_data(:, 1));
    DataPre = [DataPre; cep_data];
end

DataLabels = [zeros(size(DataPre,1), 1)];

DataPre = [DataLabels DataPre];

predict_label_work = trainedModel1.predictFcn(DataPre(:,1:26));

%-------------K=1---------------------------------------------------------
predict_label_work1 = trainedModel1.predictFcn(DataPre(:,1:26));

a=0;
for i=1:length(DataPre)
    if predict_label_work1(i,1)==DataPre(i,1)
        a=a+1;
    end
end
accuracy_frame1=a/length(DataPre)
%-------------K=3---------------------------------------------------------
predict_label_work3 = trainedModel3.predictFcn(DataPre(:,1:26));
a=0;
for i=1:length(DataPre)
    if predict_label_work3(i,1)==DataPre(i,1)
        a=a+1;
    end
end
accuracy_frame3=a/length(DataPre)
%-------------K=5---------------------------------------------------------
predict_label_work5 = trainedModel5.predictFcn(DataPre(:,1:26));