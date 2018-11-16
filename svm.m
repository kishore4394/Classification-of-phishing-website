dataset=load('dataset for training.txt');
[row_dataset col_dataset]=size(dataset);
train=dataset(1:row_dataset*0.6,:);
test=dataset((row_dataset*0.6+1):row_dataset,:);
train_label=dataset(1:row_dataset*0.6,col_dataset);
test_label=dataset((row_dataset*0.6+1):row_dataset,col_dataset);
cp=classperf(test_label);
svmStruct=svmtrain(train,train_label,'showplot',true);
classes=svmclassify(svmStruct,test,'showplot',true);
classperf(cp,classes,test);
cp.CorrectRate

