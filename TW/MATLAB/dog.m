function Y = dog(X,sigma1, sigma2, normalize)

%% Default results
Y=[];

%% Parameter checking
if nargin == 1
    sigma_one = 1;
    sigma_two = 2;
    normalize = 1;
elseif nargin == 2
    if isempty(sigma1)
        sigma1 = 1;
    end
    
    sigma_one = sigma1;
    sigma_two = 2*sigma1;
    normalize = 1;
elseif nargin == 3
    if isempty(sigma1)
        sigma1 = 1;
    end
    
    if isempty(sigma2)
       sigma2 = 2; 
    end
    
    if ~(length(sigma1)==1 && length(sigma2)==1)
       disp('Error: The parameters sigma1 and sigma2 need to be scalars.');
       return;
    else
        sigma_one = sigma1;
        sigma_two = sigma2;
    end 
    normalize = 1;
elseif nargin == 4   
    if isempty(sigma1)
        sigma1 = 1;
    end
    
    if isempty(sigma2)
       sigma2 = 2; 
    end
        sigma_one = sigma1;
        sigma_two = sigma2; 
        
     if ~(normalize==1 || normalize==0)
        disp('Error: The fourth parameter can only be 0 or 1.');
        return;
     end
elseif nargin > 4
   disp('Error: The function takes at most four parameters.');
   return;
end



%% Init. operations
[a,b]=size(X);
F1 = fspecial('gaussian',2*ceil(3*sigma_one)+1,sigma_one);
F2 = fspecial('gaussian',2*ceil(3*sigma_two)+1,sigma_two);
X1=normalize8(X); 

%% Filtering
XF1  = (imfilter(X1,F1,'replicate','same'));
XF2  = (imfilter(X1,F2,'replicate','same'));

Y = XF1-XF2;

%% postprocessing
if normalize ~= 0
    [Y, dummy] =histtruncate(Y, 0.2, 0.2);
    Y=normalize8(Y);  
end