imSize = 100;                                     % image size n x n
sigma = 8;                                    % gaussian standard deviation in pixels
trim = 0.005;                                     % trim off gaussian values smaller than this

X = 1:imSize;                                     % X is a vector from 1 to imSize
X0 = (X/imSize)-0.5;                              % rescale X to -0.5 to 0.5

[Xm, Ym] = meshgrid(X0,X0);                       % 2D squre matrix
                                                                            
s = sigma / imSize;                               % s scaled to fit the image size

Gauss = exp(-(((Xm.^2)+(Ym.^2)) ./ (2* s^2)));    % formula for 2D gaussian
                                                  % f(x,y) = A*exp(-((x-x0)^2/2*thetax^2 
                                                  %           + (y-y0)^2/2*thetay^2))
                                                  % Here A=1, x0=y0=0, thetax=thetay=s

Gauss(Gauss < trim) = 0;                          % Trim the values below 0.005
GF = fspecial('gaussian', [5,5], 1) ;             % hsize = (4*sigma+1)x(4*sigma+1)    

GN_1 = randn(imSize,imSize)*sqrt(0.04*0.04);      % Gaussian noise 4%
GN_4 = randn(imSize*2, imSize*2)*sqrt(0.04*0.04);
GN_16= randn(imSize*4, imSize*4)*sqrt(0.04*0.04);

Gauss_1 = Gauss + GN_1;                           % Add noise to Gauss Image
Gauss_4 = repmat(Gauss,2,2) + GN_4;
Gauss_16= repmat(Gauss,4,4) + GN_16;


%-------------------------Original Images without tranlation---------------
% 1. One Gaussian core
Gauss_1 = imfilter(Gauss_1, GF);                  % Image prefiltering
imwrite(Gauss_1, '1_Core.bmp','bmp');
figure
imshow(Gauss_1);
Gauss_1_Streched = imresize(Gauss_1, [100/2 100*2]);
figure
imshow(Gauss_1_Streched);
imwrite(Gauss_1_Streched, '1_Core_Streched.bmp','bmp');

% 2. 4 Gaussian cores
Gauss_4 = imfilter(Gauss_4, GF); 
imwrite(Gauss_4, '4_Core.bmp','bmp');
figure
imshow(Gauss_4);
Gauss_4_Streched = imresize(Gauss_4, [200/2 200*2]);
figure
imshow(Gauss_4_Streched);
imwrite(Gauss_4_Streched, '4_Core_Streched.bmp','bmp');

% 3. 16 Gaussian cores
Gauss_16 = imfilter(Gauss_16, GF); 
imwrite(Gauss_16, '16_Core.bmp', 'bmp');
figure
imshow(Gauss_16);
Gauss_16_Streched = imresize(Gauss_16, [400/2 400*2]);
figure
imshow(Gauss_16_Streched);
imwrite(Gauss_16_Streched, '16_Core_Streched.bmp','bmp');