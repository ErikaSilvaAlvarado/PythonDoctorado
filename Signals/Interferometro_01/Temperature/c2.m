close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%% Carga de datos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for o=29:56
P1 = csvread(['W00' ,num2str(o),'.csv'],30,0);

x=P1(:,1);
y=P1(:,2);

hold on
plot(x,y,'Color', [ rand, rand, rand ],'LineWidth',1.5 ); 
 ax = gca;
 box on
 ax.LineWidth = 1.2;

%  legend('0\circC1','30\circC','38\circC','50\circC','60\circC','70\circC','79\circC','90\circC','100\circC','111\circC','120\circC','130\circC','139\circC','149\circC','159\circC','170\circC','181\circC','190\circC','199\circC','210\circC','120\circC','223\circC','230\circC','240\circC','249\circC','260\circC','270\circC','279\circC','289\circC','300\circC');

 legend('300\circC1','286\circC','280\circC','270\circC','260\circC','250\circC','240\circC','232\circC','220\circC','204\circC','200\circC','190\circC','179\circC','171\circC','162\circC','151\circC','141\circC','131\circC','121\circC','112\circC','100\circC','90\circC','80\circC','71\circC','61\circC','51\circC','41\circC','32\circC');
title('Temperature');
axis([1500,1600,-100,-20])

LineWidth = 10;
set(gca,'FontSize',20)
xlabel('Wavelength(nm)');
ylabel('Output Power(dBm)');

hold off

end
 
