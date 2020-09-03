%load the dataset and label dataset
rootfolder='C:\Users\carlo\Desktop\Carlos\Universidad\MII\segundo\TFM\CNN\DATASETS\DATASET_CNN';
categories={'Car','Motorbike','Face'};
imds = imageDatastore(fullfile(rootfolder, categories),'Labelsource','foldernames');
%We have to resize the dataset
inputSize = [80 60];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);

%Split dataset into training and validation
[imdsTrain,imdsValidation]=splitEachLabel(imds,0.75);
imdsTrain=shuffle(imdsTrain);
imdsValidation=shuffle(imdsValidation);
m=size(imdsTrain.Files,1);
Y_train=zeros(size(imdsTrain.Files,1),3);
for i=1:m
    A=readimage(imdsTrain,i);
    X_train(i,:)=A(:);
    switch imdsTrain.Labels(i)
        case 'Motorbike'
            Y_train(i,2)=1;
        case 'Car'
            Y_train(i,1)=1;
        case 'Face'
            Y_train(i,3)=1;
    end
end

m_val=size(imdsValidation.Files,1);
Y_val=zeros(size(imdsValidation.Files,1),3);

for i=1:m_val
    A=readimage(imdsValidation,i);
    X_val(i,:)=A(:);
    switch imdsValidation.Labels(i)
        case 'Motorbike'
            Y_val(i,2)=1;
        case 'Car'
            Y_val(i,1)=1;
        case 'Face'
            Y_val(i,3)=1;
    end
end


