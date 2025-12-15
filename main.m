clear 
warning off
seed = 0;
rng('default');
rng(seed);
param.seed = seed;

% parameters setting
param.maxIter = 10;
param.bit_set = [16 32 64 128];
param.alpha = 1e-2; 
param.beta = 1e-4; 
param.delta = 1e-2; 
param.theta0 = 1e-5; 
param.theta1 = 1e-5; 
param.theta2 = 1e-5; 

load("D:\Code\Matlab\Dataset\Name1\flickr.mat");
inx = randperm(size(databaseL,1));
XDatabase = XDatabase(inx,:); 
YDatabase = YDatabase(inx,:);
databaseL = databaseL(inx,:);

%% centralization
XTestC = bsxfun(@minus, XTest, mean(XDatabase, 1)); XDatabaseC = bsxfun(@minus, XDatabase, mean(XDatabase, 1));
YTestC = bsxfun(@minus, YTest, mean(YDatabase, 1)); YDatabaseC = bsxfun(@minus, YDatabase, mean(YDatabase, 1));

%% kernelization 
n_anchors = 500; 
[n, ~] = size(YDatabase);
anchor_image = XDatabase(randsample(n, n_anchors),:); 
anchor_text = YDatabase(randsample(n, n_anchors),:);
XDatabaseK = RBF_fast(XDatabase',anchor_image');
YDatabaseK = RBF_fast(YDatabase',anchor_text');

%% load dataset
dataset.XTest = XTestC;
dataset.YTest = YTestC;
dataset.XDatabase = XDatabaseC; 
dataset.YDatabase = YDatabaseC;
dataset.XDatabaseK = XDatabaseK;
dataset.YDatabaseK = YDatabaseK;
dataset.testL = testL;
dataset.databaseL = databaseL;


for bit_index = 1: length(param.bit_set)
    param.bit = param.bit_set(bit_index);
    ACSCH(dataset, param);
end
