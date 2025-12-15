function ACSCH(dataset, param)

X1 = dataset.XDatabase; % n × d1
X2 = dataset.YDatabase; % n × d2
X1Test = dataset.XTest;
X2Test = dataset.YTest;
LTrain = dataset.databaseL;
LTest = dataset.testL;

maxIter = param.maxIter;
bit = param.bit;
alpha = param.alpha;
beta = param.beta;
delta = param.delta;
theta0 = param.theta0;
theta1 = param.theta1;
theta2 = param.theta2;

n = size(X1,1);
X1 = X1'; X2 = X2';

% Initialize Hash codes
sel_sample = X1(:,randsample(n, 2000),:);
[pcaW, ~] = eigs(cov(sel_sample'), bit); 
H = pcaW'*X1;
B = sign(H);
B(B==0) = -1;

% Compute pseudolabels
L = KmeansCluster(dataset.XDatabaseK', dataset.YDatabaseK'); % c × n
V = ALS(L);
V = V';

F1 = X1 * X1';
F2 = X2 * X2';
F3 = V * V';

tic;
for iter = 1:maxIter

    W1 = theta1 * H * X1' / (theta1 * F1 + delta * eye(size(X1,1)));
    W2 = theta2 * H * X2' / (theta2 * F2 + delta * eye(size(X2,1)));
    W0 = theta0 * H * V' / (theta0 * F3 + delta * eye(size(V,1)));

    O = theta1 * W1 * X1 + theta2 * W2 * X2 + theta0 * W0 * V ...
        + alpha * B + beta * bit * B * L' * L;
    O = O';
    Temp = O'*O-1/n*(O'*ones(n,1)*(ones(1,n)*O));
    [~,Lmd,RR] = svd(Temp); %clear Temp
    idx = (diag(Lmd)>1e-4);
    R = RR(:,idx); R_ = orth(RR(:,~idx));
    K = (O-1/n*ones(n,1)*(ones(1,n)*O)) *  (R / (sqrt(Lmd(idx,idx))));
    K_ = orth(randn(n,bit-length(find(idx==1))));
    H = sqrt(n)*[K K_]*[R R_]';
    H = H';  

    B = sign(alpha * H + beta * bit * H * L' * L);

    % % object value
    % objectvalue = theta1 * norm(H - W1 * X1,'fro') ^ 2 + delta * norm(W1,'fro') ^ 2 ...
    %               + theta2 * norm(H - W2 * X2,'fro') ^ 2 + delta * norm(W2,'fro') ^ 2 ...
    %               + theta0 * norm(H - W0 * V,'fro') ^ 2 + delta * norm(W0,'fro') ^ 2 ...
    %               + alpha * norm(H - B,'fro') ^ 2  ...
    %               + beta * norm(H' * B - bit* (L' * L),'fro') ^ 2 ; 
    % 
    % fprintf("objectvalue:%.f, iter:%.f \n", objectvalue, iter);


    
end

train_time = toc;
% fprintf("train_time: %.2f\n", train_time);

%% Hash codes
sigma = 1e-3;
BX1Train = double(B > 0)';
BX2Train = double(B > 0)';
T1= B * X1' /(X1 * X1' + sigma * eye(size(X1,1)));
T2= B * X2' /(X2 * X2' + sigma * eye(size(X2,1)));
BX1Test = double(X1Test*T1'>0);
BX2Test = double(X2Test*T2'>0);


%% Compute mAP
Recall_set = 100;%[5,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000];

DHamm = pdist2(BX1Test, BX2Train,'hamming');
[~, orderH] = sort(DHamm, 2);
for Recall = Recall_set
    ImgToTxt = mAP(orderH', LTrain, LTest, Recall); 
    fprintf("bits:%.d, Recall:%.d, ImgToTxt:%.4f \n", bit, Recall, ImgToTxt);
end

% [Image_to_Text_precision, Image_to_Text_recall] = precision_recall(orderH',LTrain, LTest);

DHamm = pdist2(BX2Test, BX1Train,'hamming');
[~, orderH] = sort(DHamm, 2);
for Recall = Recall_set
    TxtToImg = mAP(orderH', LTrain, LTest, Recall);  
    fprintf("bits:%.d, Recall:%.d, TxtToImg:%.4f \n", bit, Recall, TxtToImg);
end

% [Text_to_Image_precision, Text_to_Image_recall] = precision_recall(orderH',LTrain, LTest);

% fprintf('ACSCH %d bits, alpha:%.5g, beta:%.5g, delta: %.5g,  theta0:%.5g, theta1&2: %.5g, ImgToTxt: %.4f, TxtToImg: %.4f \n', bit, alpha, beta, delta, theta0, theta1, ImgToTxt, TxtToImg);
% fprintf('ACSCH %d bits, alpha:%.5g, beta:%.5g, delta: %.5g, ImgToTxt: %.4f, TxtToImg: %.4f \n', bit, alpha, beta, delta, ImgToTxt, TxtToImg);
% fprintf('%.4f, %.4f \n', ImgToTxt, TxtToImg);

end

