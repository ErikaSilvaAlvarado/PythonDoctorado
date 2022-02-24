close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%% Carga de datos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for o=0:27
P1 = csvread(['W00' ,num2str(o),'.csv'],30,0);

x=P1(:,1);
y=P1(:,2);

hold on
       plot(x,y,'Color', rand(1,3),'LineWidth',2 ); 
%       plot(x,y,'b','LineWidth',2 );
 ax = gca;
 box on
 ax.LineWidth = 1.2;

%  legend('30\circC','38\circC','50\circC','60\circC','70\circC','79\circC','90\circC','100\circC','111\circC','120\circC','130\circC','139\circC','149\circC','159\circC','170\circC','181\circC','190\circC','199\circC','210\circC','120\circC','223\circC','230\circC','240\circC','249\circC','260\circC','270\circC','279\circC','289\circC','300\circC');
% title('Temperature');
  axis([1545,1560,-90,-10,])
LineWidth = 10;
set(gca,'FontSize',20)
xlabel('Wavelength(nm)');
ylabel('Output Power(dBm)');

hold off

end
 
