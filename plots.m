clear;
file_P = "D:\labs\year3\Project\codes_v2\total_P_new40.csv";
file_Q = "D:\labs\year3\Project\codes_v2\total_Q_new40.csv";
file_dec = "D:\labs\year3\Project\codes_v2\decomposition_new10.csv";
%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: D:\labs\year3\Project\codes\total_P.csv
%
% Auto-generated by MATLAB on 09-Feb-2022 00:06:59

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = "P";
opts.VariableTypes = "double";

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
tbl = readtable(file_P, opts);

%% Convert to output type
P = tbl.P;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: D:\labs\year3\Project\codes\total_Q.csv
%
% Auto-generated by MATLAB on 09-Feb-2022 00:11:53

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = "Q";
opts.VariableTypes = "double";

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
tbl = readtable(file_Q, opts);

%% Convert to output type
Q = tbl.Q;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: D:\labs\year3\Project\codes\decomposition.csv
%
% Auto-generated by MATLAB on 09-Feb-2022 00:14:22

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 6);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["label0", "label1", "label2", "label3", "label4", "label5"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
tbl = readtable(file_dec, opts);

%% Convert to output type
label0 = tbl.label0;
label1 = tbl.label1;
label2 = tbl.label2;
label3 = tbl.label3;
label4 = tbl.label4;
label5 = tbl.label5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: D:\labs\year3\Project\codes\predictions.csv
%
% Auto-generated by MATLAB on 09-Feb-2022 14:46:49

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 6);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["pred0", "pred1", "pred2", "pred3", "pred4", "pred5"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
tbl = readtable("D:\labs\year3\Project\codes_v2\predictions.csv", opts);

%% Convert to output type
pred0 = tbl.pred0;
pred1 = tbl.pred1;
pred2 = tbl.pred2;
pred3 = tbl.pred3;
pred4 = tbl.pred4;
pred5 = tbl.pred5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear temporary variables
clear opts tbl

figure; plot(P);
xlabel('sample') ;
ylabel('P') ;
title('total P');
grid on

figure; plot(Q);
xlabel('sample') ;
ylabel('Q') ;
title('total Q');
grid on

labels = [label0 label1 label2 label3 label4 label5];
figure; plot(labels);
%plot(label0);
%plot(label1); hold on;
%plot(label2);
%plot(label3);
%plot(label4);
%plot(label5);
xlabel('sample') ;
ylabel('percentage') ;
title('decompocition (all data)');
legend('label0', 'label1', 'label2', 'label3', 'label4', 'label5');
grid on; hold off;

predictions = [pred0 pred1 pred2 pred3 pred4 pred5];
figure; plot(predictions);
%plot(pred0);
%plot(pred1); hold on;
%plot(pred2);
%plot(pred3);
%plot(pred4);
%plot(pred5);
xlabel('sample') ;
ylabel('percentage') ;
title('prediction (100 predictions)');
legend('pred', 'pred1', 'pred2', 'predl3', 'pred4', 'pred5');

grid on; hold off;


first_pred = 5876;
pred_num = 2000;
ev0_comp = label0(first_pred: first_pred+pred_num-1);
ev1_comp = label1(first_pred: first_pred+pred_num-1);
ev2_comp = label2(first_pred: first_pred+pred_num-1);
ev3_comp = label3(first_pred: first_pred+pred_num-1);
ev4_comp = label4(first_pred: first_pred+pred_num-1);
ev5_comp = label5(first_pred: first_pred+pred_num-1);

ev_comp = [ev0_comp ev1_comp ev2_comp ev3_comp ev4_comp ev5_comp];
figure; plot(ev_comp);
%plot(ev0_comp);
%plot(ev1_comp); hold on;
%plot(ev2_comp);
%plot(ev3_comp);
%plot(ev4_comp);
%plot(ev5_comp);
xlabel('sample') ;
ylabel('percentage') ;
title('real values (100 samples)');
legend('ev0comp', 'ev1comp', 'ev2comp', ' ev3comp', 'ev4comp', 'ev5comp');
grid on; hold off;

diff0= ev0_comp - pred0;
diff1= ev1_comp - pred1;
diff2= ev2_comp - pred2;
diff3= ev3_comp - pred3;
diff4= ev4_comp - pred4;
diff5= ev5_comp - pred5;

differences = [diff0 diff1 diff2 diff3 diff4 diff5];
figure; plot(differences);
%plot(diff0);
%plot(diff1); hold on;
%plot(diff2);
%plot(diff3);
%plot(diff4);
%plot(diff5);
xlabel('sample') ;
ylabel('percentage') ;
title('errors (100 samples)');
legend('error0', 'error1', 'error2', ' error3', 'error4', 'error5');
grid on; hold off;
