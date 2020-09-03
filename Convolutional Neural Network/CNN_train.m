%compute Covolutional Neuronal Network
%first we have to load the dataset.
rootfolder='DATASETS\DATASET_CNN';
categories={'Car','Motorbike','Face'};
imds = imageDatastore(fullfile(rootfolder, categories),'Labelsource','foldernames');

%dividimos el dataset en entrenamiento y validación
[imdsTrain,imdsValidation]=splitEachLabel(imds,0.75);
%Definimos los parametros de nuestra red convolucional
K1=5; %tamaño del filtro de la primera capa
S1= [1 1]; %stride de la primera capa
N1=8; %numero de filtros de la primera capa convolucional
P1=3; %tamaño del pooling
K2=K1;
S2=S1;
N2=16;
P2=2;
K3=3;
S3=S1;
N3=32;
...
    
%Creamos un augmented Image Data Store
imageSize=[80 60 3];
auimdsTrain=augmentedImageDatastore(imageSize,imdsTrain);
auimdsValidation=augmentedImageDatastore(imageSize,imdsValidation);

%El parametro padding lo modificaremos directamente sobre la red
%Definimos la arquitectura de la red convolucional

layers=[
    imageInputLayer(imageSize)
    convolution2dLayer(K1,N1,'padding','same','stride',S1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(P1,'stride',1)
    
    convolution2dLayer(K2,N2,'stride', S2)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(P1,'stride',1)
    
    convolution2dLayer(K3,N3,'stride', S3,'padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(K3,N3,'stride', S3,'padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(3)
    softmaxLayer()
    classificationLayer()]


%Definimos las opciones del entrenamiento de la red

options = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',4,...
    'Shuffle','every-epoch','ValidationData',auimdsValidation, ...
    'ValidationFrequency',30,'Verbose',true,'Plots','training-progress',...
    'MiniBatchSize',64,'ExecutionEnvironment','cpu','ValidationPatience',Inf)
net=trainNetwork(auimdsTrain,layers,options);
    
    