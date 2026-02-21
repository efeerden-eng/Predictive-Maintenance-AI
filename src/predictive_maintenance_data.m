%% PREDICTIVE MAINTENANCE PROJECT

%% 1. DATA LOADING AND CLEANING
% Load the industrial data from Bosch
data = readtable('ai4i2020.csv'); 

% Remove ID columns because serial numbers are not related to physics
data.UDI = []; 
data.ProductID = [];

% Set professional column names for easier coding
data.Properties.VariableNames = {'Type', 'AirTemp', 'ProcTemp', 'Speed', 'Torque', 'ToolWear', 'Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'};

%% 2. FEATURE ENGINEERING
% Calculate the heat difference between machine and environment
data.TempDiff = data.ProcTemp - data.AirTemp;

% Calculate Mechanical Power (Torque * Speed) to see the engine load
data.Power = data.Torque .* (2 * pi * data.Speed / 60);

% Convert machine quality (L, M, H) into numbers for the AI
data.TypeNumeric = double(categorical(data.Type)); 

%% 3. EXTERNAL DATA SPLITTING
% Select our inputs (sensors) and our target (Failure)
X_table = data(:, {'TypeNumeric', 'AirTemp', 'ProcTemp', 'Speed', 'Torque', 'ToolWear', 'TempDiff', 'Power'});
X = table2array(X_table); 
y = data.Failure;

% Split data: 80% for learning process, 20% for testing
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

%% 4. ARTIFICIAL INTELLIGENCE SETUP
% Create the brain
net = patternnet([10 8]);

% Manual split settings: Use the 80% for learning and validation
% We set testRatio to 0 because we already have our own 20% test set
net.divideParam.trainRatio = 80/100; 
net.divideParam.valRatio = 20/100;   
net.divideParam.testRatio = 0/100;    

%% 5. PENALTY SYSTEM AND TRAINING
% Since failures are rare, we give a 20x penalty for missing a failure
weights = ones(size(y_train));
weights(y_train == 1) = 20; 

% Start training using transposed (') data and penalty weights
[net, tr] = train(net, X_train', y_train', [], [], weights');

%% 6. PERFORMANCE EVALUATION
% Testing the AI with the 20% data it has never seen before
y_pred = net(X_test');
plotconfusion(y_test', y_pred);
%% 7. VISUALIZATION & ANALYSIS
% After training, we evaluate our model with different perspectives

%  A. Confusion Matrix
% This shows where our AI succeeded and where it gave false alarms
y_pred = net(X_test');
plotconfusion(y_test', y_pred);
title('20% Unseen Data Results');

%  B. Performance Plot
% Checking if we have "overfitting" or if the model learned properly
figure;
plotperform(tr);
% Note: If Validation line stays close to Training, it means our [10 8] structure is stable.

%  C. Error Histogram (Deep Error Analysis)
% We check the distribution of errors to see the precision of the AI
figure;
error_test = y_test' - y_pred;
ploterrhist(error_test, 'Bins', 20);
title('Error Distribution');

%  D. ROC Curve
% This proves the AI is a professional decision maker
[Xroc, Yroc, Troc, AUC] = perfcurve(y_test, y_pred, 1);
figure('Color', 'w');
plot(Xroc, Yroc, 'LineWidth', 3);
grid on;
xlabel('False Positive Rate (False Alarms)');
ylabel('True Positive Rate (Success)');
title(['ROC Curve - Area Under Curve (AUC): ', num2str(AUC)]);

%% 8. REAL-WORLD DEPLOYMENT TEST
% I will test the AI with a random scenario I created
% Data: [Type, AirTemp, ProcTemp, Speed, Torque, ToolWear, TempDiff, Power]
my_machine = [2, 100, 21, 80, 55, 18, 10, 1612]; 

prediction = net(my_machine');

fprintf("TEST RESULT");
if prediction > 0.5
    fprintf('WARNING: AI predicts a FAILURE for this machine!');
else
    fprintf('SYSTEM OK: This machine is working perfectly.');
end
fprintf('-------------------\n');