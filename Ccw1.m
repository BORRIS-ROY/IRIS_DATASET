load fisheriris
X = meas;
Y = species;
gscatter(X(:,1) , X(:,2),species , 'rgb');
Z = randperm(150);%Randomising the indices.
bus = 1;
%Training Data
xTrain = X(Z(bus:50),:);
yTrain = Y(Z(bus:50),:);
gscatter(xTrain(:,1) , xTrain(:,2),yTrain , 'rgb');
%Test Data
xTest = X(Z(50+bus:100),:);
yTest = Y(Z(50+bus:100),:);
%Validation Data
xValidate = X(Z(100+bus:150),:);
yValidate = Y(Z(100+bus:150),:);
fprintf('3-fold cross validation: fold 1')
good_rate = 100; %intialize best rate to 100
k_Best = 0; %intialize best value of K to 0
for k=1:2:length(xTrain) %choose K in odd numbers btn 1-75
    knn1 = fitcknn(xTrain,yTrain,'NumNeighbors',k);
    yPred = predict(knn1,xTest);
    hold on;
    gscatter(xTest(:,1), xTest(:,2), yPred,'rgb');%Visualise classification results.
    error = 0;
    for i=1:length(yTest)
        if ~isequal(yPred(i), yTest(i))
            error = error+1;
        end
    end
    error_rate = error/50;
    
    if error_rate < good_rate
        k_Best = k;
        good_rate = error_rate;
    end
end
mean_error = good_rate %Print best rate
k_Best    %Print best K
fprintf('3-fold cross validation: fold 2')
good_rate = 100; %intialize best rate to 100
k_Best = 0; %intialize best value of K to 0
for k=1:2:length(xTrain) %choose K in odd numbers btn 1-75
    knn1 = fitcknn(xTrain,yTrain,'NumNeighbors',k);
    yPred2 = predict(knn1,xValidate);
    hold on;
    gscatter(xValidate(:,1), xValidate(:,2), yPred2,'rgb');%Visualise classification results.
    error = 0;
    for i=1:length(yValidate)
        if ~isequal(yPred2(i), yValidate(i))
            error = error+1;
        end
    end
    error_rate = error/50;
    
    if error_rate < good_rate
        k_Best = k;
        good_rate = error_rate;
    end
end
error_mean = good_rate %Print best rate
k_Best    %Print best K


