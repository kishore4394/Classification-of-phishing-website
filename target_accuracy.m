dataset=load('dataset for training.txt');
[row_dataset col_dataset]=size(dataset);
train=dataset(1:row_dataset*0.6,1:col_dataset-1);
test=dataset((row_dataset*0.6+1):row_dataset,1:col_dataset-1);
train_label=dataset(1:row_dataset*0.6,col_dataset);
test_label=dataset((row_dataset*0.6+1):row_dataset,col_dataset);
nb=NaiveBayes.fit(train,train_label,'Distribution','mvmn');
predicted_label=predict(nb,test);
difference_label=test_label-predicted_label;
[row_test col_test]=size(test);
accuracy=(sum(difference_label(:)==0))/row_test*100