close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%% Carga de datos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for o=63:71
P1 = csvread(['W00' ,num2str(o),'.csv'],30,0);

x=P1(:,1);
y=P1(:,2);

hold on
plot(x,y,'Color', [ rand, rand, rand ],'LineWidth',2 ); 
 ax = gca;
 box on
 ax.LineWidth = 1.2;

  legend('30\circC','40\circC','50\circC','63\circC','71\circC','85\circC','95\circC','107\circC','120\circC');

%  legend('300\circC','286\circC','280\circC','270\circC','260\circC','250\circC','240\circC','232\circC','220\circC','204\circC','200\circC','190\circC','179\circC','171\circC','162\circC','151\circC','141\circC','131\circC','121\circC','112\circC','100\circC','90\circC','80\circC','71\circC','61\circC','51\circC','41\circC','32\circC');
title('Temperature');
axis([1520,1580,-80,-10,])

LineWidth = 10;
set(gca,'FontSize',20)
xlabel('Wavelength(nm)');
ylabel('Output Power(dBm)');

hold off

end
 
