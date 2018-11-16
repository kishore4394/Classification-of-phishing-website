dataset=load('dataset for training.txt');
resultant_dataset=[dataset(:,8)];
[row_dataset col_dataset]=size(dataset);
for j=1:col_dataset-1
for k=1:col_dataset-1
new_dataset=[resultant_dataset dataset(:,k)];
[row_new_dat col_new_dat]=size(new_dataset);
train=new_dataset(1:row_new_dat*0.6,:);
test=new_dataset((row_new_dat*0.6+1):row_new_dat,:);
train_label=dataset(1:row_dataset*0.6,col_dataset);
test_label=dataset((row_dataset*0.6+1):row_dataset,col_dataset);
nb=NaiveBayes.fit(train,train_label,'Distribution','mvmn');
predicted_label=predict(nb,test);
difference_label=test_label-predicted_label;
[row_test col_test]=size(test);
accuracy_k=(sum(difference_label(:)==0))/row_test*100;
accuracy_set(k,1)=[accuracy_k];
accuracy_final_set=[accuracy_set(:,1)];
end
[y,i]=max(accuracy_final_set);
resultant_dataset=[resultant_dataset dataset(:,i)];
set(j,:)=[y i];
disp(set)
end