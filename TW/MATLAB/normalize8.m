function Y=normalize8(X,mode);

%% default return value
Y=[];

%% Parameter check
if nargin==1
    mode = 1;
end

%% Init. operations
X=double(X);
[a,b]=size(X);

%% Adjust the dynamic range to the 8-bit interval
max_v_x = max(max(X));
min_v_x = min(min(X));

if mode == 1
    Y=ceil(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b))))*255);
elseif mode == 0
    Y=(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b)))));
else
    disp('Error: Wrong value of parameter "mode". Please provide either 0 or 1.')
end