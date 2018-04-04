tic
close all
clear
clc
%y=load('E:\em one 20171227\data\raw_fig4\24def_0_8_n2_FFTCC-ICGN_BC_O8_Data.csv');
y=load('Sample 14 L3 Amp0_8.3_Data.csv');
v8=y(:,4);%
%[h1,w1]=size(v8);
h2=34;
w2=326;
v88=reshape(v8,w2,h2);
figure;
surf(v88'); colormap(jet);
shading interp;
M=9;%strain window 2*M+1
GS=5;%grid space
SuSi=10;%subset size SuSi*2+1
MarSi=200;% margin space
%uvalue=0.000000006;
accuracyorder=8;% accuracy order O8
%coe1=-1*M:M;

for i=M+1:w2-M
    for j=M+1:h2-M
        st0=0;
        for ii=-1*M:M
            for jj=-1*M:M
                st0=st0+ii*v88(i+ii,j+jj);%【】
            end
        end
        st1(i-M,j-M)=3/(GS*M*(M+1)*(2*M+1)^2)*st0*10^(0);
    end
end

st1 = st1*10^6;
figure;
surf(st1'); colormap(jet);
shading interp;
xlabel('X (pixels)');
ylabel('Y (pixels)');
zlabel('Calculated Strain (με)');

maxy=max(max(st1))
miny=min(min(st1))
%u4=u(:,4);%加载位移数据。
%u44=reshape(u4,145,145);%需要对应改动大小。
%u44=u4;%需要对应改动大小。
%=u44;

st1one = st1';

y=load('Sample 14 Commanded Displacement.csv');
v8=y(:,5);

[h,w]=size(st1);
weiz=1:h;
hvalue=weiz;
for ii=1:h
    hvalue(ii)=v8(1+accuracyorder+SuSi+MarSi+(ii-1+M)*GS);    
end
jzvalue=st1;

for jj=1:w
    jzvalue(:,jj)=hvalue;
end
jzvalue1=jzvalue;
figure;surf(jzvalue1'); colormap(jet);
shading interp;
xlabel('X (pixels)');
ylabel('Y (pixels)');
zlabel('Pre-set Strain (με)');

jzvalue1one = jzvalue1';
figure; plot(jzvalue1one(1,:)); hold on
plot(st1one(1,:));
xlabel('X (pixels)');
ylabel('Strain (με)');
legend('Pre-set Strain','Calculated Strain');

st2=st1-jzvalue1;%
syst=mean2(st2)%
std2=std2(st1)%除以n-1
%std3=sqrt(sum(sum((st1-mean2(st1)).^2))/(h*w-1))

rmse=sqrt(mean2(st2.^2))%除以n  root mean squre error
toc