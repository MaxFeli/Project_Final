%% Data augmentation

dir_name = 'D:\Users\Info\Desktop\Max Felicitas Property\Scuola\Unipi\Magistrale\Laboratorio di Meccanica e Meccatronica\Project\Project_v4\dataset';
folder = dir(dir_name);

tic;
for i=3:length(folder)
    if (~folder(i).isdir)
        % do things to augment images
        file = folder(i).name;
        aug1(dir_name, 1, file); 
        aug2(dir_name, 2, file);
    end
end
toc;

f = msgbox('Operation Complete','Success');

%% Functions to augment data

function aug1(fold_name, index, file)
    img = imread(strcat(fold_name,'\',file));
    scale = rand()+1;             % scale factor
    angle = (10*rand()-5)*pi/180; % rotation angle
    tx = -40*rand()+20;           % x translation
    ty = -40*rand()+20;           % y translation
    a = 1;                        % -1 -> reflection, 1 -> no reflection

    sc = scale*cos(angle);
    ss = scale*sin(angle);

    T = [   sc   -ss  0;
            a*ss  a*sc  0;
            tx    ty  1];
    
    t_sim = affine2d(T);
    img = imwarp(img,t_sim);
    img = imresize(img,[480 640]);
    newStr = erase(file,'.png');
    name = strcat(fold_name,'\',newStr,'_',string(index),'.png');
    imwrite(img, name);
end

function aug2(fold_name, index, file)
    img = imread(strcat(fold_name,'\',file));
    scale = 1;               % scale factor
    angle = (5*rand()-2.5)*pi/180; % rotation angle
    tx = 0;                  % x translation
    ty = 0;                  % y translation
    a = 1;                   % -1 -> reflection, 1 -> no reflection

    sc = scale*cos(angle);
    ss = scale*sin(angle);

    T = [   sc   -ss  0;
            a*ss  a*sc  0;
            tx    ty  1];
        
    t_sim = affine2d(T);
    img = imwarp(img,t_sim);
    img = imgaussfilt(img, rand()+2);
    img = imresize(img,[480 640]);
    newStr = erase(file,'.png');
    name = strcat(fold_name,'\',newStr,'_',string(index),'.png');
    imwrite(img, name);
end