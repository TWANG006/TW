RGB = [0,0,0];                  % Color of the circle
I(1:4000,1:4000) = 255;         % Original white image
I = uint8(I);

num_circle = 65*65;

r = randi([-25,25],num_circle,2);    % Generate 66x66 random integers
pos_circle = zeros(num_circle,3);    % positions of circles

for i = 1:num_circle
    pos_circle(i,1) = 61 + mod(i-1,65)*60;
    pos_circle(i,2) = 61 + (fix((i-1)/65)+1)*60;
    
    pos_circle(i,1) = pos_circle(i,1)+r(i,1);
    pos_circle(i,2) = pos_circle(i,2)+r(i,2);
    
    pos_circle(i,3) = 22.5;
end

% Insert circles into the image
I = insertShape(I, 'FilledCircle', pos_circle, 'Color', [0,0,0]);

% Scale the intensity values from 0-255 to 30-255
I = double(I)/255*(255-30);
I = uint8(I);
I = rgb2gray(I);
I = imcomplement(I);


% Antialiasing & Downsampling
II = imresize(I, .1, 'Antialiasing',false);

% Gaussian Blurring
G1=fspecial('gauss', [5,5], 1);
II = imfilter(II,G1);
figure,imshow(II);

imwrite(II,'img.bmp');
