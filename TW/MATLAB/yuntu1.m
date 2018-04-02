clear
clc
close all
%单次；
tic
%修改；直接求解；
%%%%%%%%%%%%%%%%%%%%%%%程序输入部分%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('E:\total error20170508\case4\em4_yuan4_7_1quan.mat');
%%
A=-1*flipdim(x,1);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y1=[608:5:808];
x1=[592:5:792];
[Y,X,Z]=griddata(y1,x1,A,linspace(608,808,100)',linspace(592,792,100),'v4');
figure
% contourf(Y,X,Z,20)
pcolor(Y,X,Z);
shading interp %伪彩色图

set(gca,'FontSize',15);
%%

 set(gca,'FontName','Times New Roman' ,'FontSize',15);
set(gca,'FontName','Times New Roman' ,'FontSize',15);
 xlim([608 808]) ;
 set(gca,'XTick',608:50:808);
 set(gca,'XTickLabel',{'608','658','708','758','808'});
 xlabel('{\itX}(pixel)','FontName','Times New Roman');
 set(gca,'FontName','Times New Roman' ,'FontSize',15);
 ylim([592 792]) ;
 set(gca,'YTick',592:50:792);
 set(gca,'YTickLabel',{'592','642','692','742','792'});
 ylabel('{\itY}(pixel)','FontName','Times New Roman');

set(gca,'FontName','Times New Roman' ,'FontSize',15);
h=colorbar;
set(get(h,'Title'),'string','pixel','FontName','Times New Roman' ,'FontSize',15);
set(h,'FontName','Times New Roman' ,'FontSize',15);