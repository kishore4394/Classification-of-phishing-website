dataset=load('dataset for training.txt');
[row_dataset col_dataset]=size(dataset);
train=dataset(1:row_dataset*0.6,1:col_dataset-1);
test=dataset((row_dataset*0.6+1):row_dataset,1:col_dataset-1);
train_label=dataset(1:row_dataset*0.6,col_dataset);
test_label=dataset((row_dataset*0.6+1):row_dataset,col_dataset);
phisphing_tree=classregtree(train,train_label,'method','classification');
eval_label=eval(phisphing_tree,test);
predicted_label=str2double(eval_label);
difference_label=test_label-predicted_label;
[row_test col_test]=size(test);
accuracy_k=(sum(difference_label(:)==0))/row_test*100
view(phisphing_tree)
