%Clustering
outputFolder = fullfile('TugasBesar');%memanggil folder
rootFolder = fullfile(outputFolder, 'Dataset Sampah');
categories = {'Botol', 'Kardus', 'Plastik'};
 
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)%mengubah semua count menjadi sama dengan nilai array klaster 2
 
botol = find(imds.Labels == 'Botol', 1);
kardus = find(imds.Labels == 'Kardus', 1);
plastik = find(imds.Labels == 'Plastik', 1);
 
net = resnet50()
%figure
%plot(net)
%title('Architecture of ResNet-50')
%set(gca, 'YLim', [150 170]);
 
net.Layers(1)
net.Layers(end)
 
numel(net.Layers(end).ClassNames)
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
imageSize = net.Layers(1) .InputSize;
 
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
 
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
 
%figure
%montage(w1)
%title('First Conculational Layer Weight');
 
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
 
trainingLables = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLables, 'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
 
testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
 
predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
 
testLables = testSet.Labels;
confMat = confusionmat(testLables, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));
 
mean(diag(confMat));
RealImage = imread('TesK.jpeg');
I = rgb2gray(imread('TesK.jpeg')); 
newImage = imread(fullfile('TesK.jpeg'));
 
ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');
 
imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
 
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
 
sprintf(' Gambar masuk dalam kategori %s', label)

%===========================================
%Membersihkan Derau
%filter wiener
L = imnoise(I,'poisson');
LLLL=wiener2(L,[5,5]);
%menggunakan filter wiener untuk membersihkan derau, pada filter ini derau
%poison menampilkan hasil yang lebih baik dibandingkan dengan parameter
%lainnya

%===========================================
%Segmentasi Pada Sampah
newImage = double(rgb2gray(newImage));
Sx = [-1 0 1; -2 0 2; -1 0 1];
Sy = [-1 -2 -1; 0 0 0; 1 2 1];
Ix = conv2(newImage,Sx,'same');
Iy = conv2(newImage,Sy,'same');
m = sqrt((Ix.^2)+(Iy.^2));

%===========================================
%Operasi Morfologi
txt_gray = rgb2gray(RealImage);
hpf=[-1 -1 -1;-1 5 -1;-1 -1 -1];
J=uint8(conv2(double(txt_gray),hpf,'same'));
img = im2bw(txt_gray);
skeleton = bwskel(img);
img2 = labeloverlay(J, skeleton, 'Transparency',0.7);

%===========================================
%Transformasi

T = [0 0; 1 0; 0 1; 0.001 0; 0.02 0; 0.01 0];
xybase = reshape(randn(12,1),6,2);
t_poly = cp2tform(xybase,xybase,'polynomial',2);
t_poly.tdata = T;
I_polynomial = imtransform(RealImage,t_poly,'FillValues',.3);

%===========================================
%Transformasi Spasial
J=imadjust(RealImage,[40/255 204/255],[0/255 255/255]);

%===========================================
%Ekstraksi Fitur
J = rgb2gray(RealImage);
wavelength = 4;
orientation = 45;
[mag45,phase45] = imgaborfilt(J,wavelength,orientation);
bw45 = mag45>1000;
orientation = 135;
[mag135,phase135] = imgaborfilt(J,wavelength,orientation);
bw135 = mag135>1000;

%==================FIGURE=========================
%Image
figure,
subplot(4,3,1), imshow(readimage(imds,botol)), title("Contoh Kategori Botol");
subplot(4,3,2), imshow(readimage(imds,kardus)), title("Contoh Kategori Kardus");
subplot(4,3,3), imshow(readimage(imds,plastik)), title("Contoh Kategori Plastik");
%Segmentasi
subplot(4,3,4), imagesc(RealImage), title("Citra Asli");
subplot(4,3,5), imagesc(newImage),axis image;
subplot(4,3,6), imagesc(m),axis image,colormap gray
%Perbaikan Derau
subplot(4,3,7),imshow(RealImage), title("Citra Asli");
subplot(4,3,8),imshow(I),title("Wiener Filter");
subplot(4,3,9),imshow(LLLL),title("perbaikan derau poison");
%Operasi Morfologi
subplot(4,3,10),imshow(RealImage), title("Citra Asli");
subplot(4,3,11),imshow(img), title("binary img");
subplot(4,3,12),imshow(img2), title("Hasil skeleton");
figure,
%Transformasi
subplot(3,2,1),imshow(RealImage), title("Citra Asli");
subplot(3,2,2), imshow(I_polynomial), title('Transformasi Polynomial');
%Transformasi Spasial
subplot(3,2,3),imshow(RealImage), title("Citra Asli");
subplot(3,2,4),imhist(RealImage), title("Histogram Citra Asli");
subplot(3,2,5),imshow(J), title("Peningkatan Kontras");
subplot(3,2,6),imhist(J), title("Histogram Peningkatan Kontras");
%Ektraksi Fitur
figure
subplot(2,4,1), imshow(RealImage),title('citra metal');
subplot(2,4,3), imshow(mag45,[]),title('magnitude gabor 45*');
subplot(2,4,4), imshow(mag135,[]),title('magnitude gabor 135*');
subplot(2,4,5), imshow(phase45,[]),title('phase gabor 45*');
subplot(2,4,6), imshow(phase135,[]),title('phase gabor 135*');
subplot(2,4,7), imshow(bw45),title('biner gabor 45*');
subplot(2,4,8), imshow(bw135),title('biner gabor 135*');
