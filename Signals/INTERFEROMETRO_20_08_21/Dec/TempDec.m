clc
clear all
close all


location = 'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Giselle\INTERFEROMETRO_20_08_21\';
dirFigure = 'C:\Users\Erika\OneDrive - Universidad de Guanajuato\Giselle\INTERFEROMETRO_20_08_21\';
nameFigure = "TempDec";
file_init = 29;
file_end = 56;
NumberFigure = 0;
temp = [300 286 280 270 260 250 240 232 220 204 200 190 179 176 162 151];
aux = num2str(temp');

[lambda,yall] = ReadFolder(location, file_init,file_end);
sizeYall= size(yall);
NS = sizeYall(2)


%Plot 
NumberFigure = NumberFigure+1;
figure ( NumberFigure)
plotParametric(lambda,yall,NS)
hold on,
yRange = [-80 -30]
line(1550*ones(1,100),linspace(yRange(1), yRange(end)))

ax = gca % current axes
ax.FontSize=8; % axis number size (IEEE )

%Style
lgd = legend(aux)
lgd.Title.String = 'Temperature (°C)';
lgd.Box='off' 

xlabel('Wavelength (nm)','FontSize',10);
xRange = [lambda(1) lambda(end)];
xlim([xRange])
dx=10;
xTicks = xRange(1): dx: xRange(2);
yRange = [-80 -30]
ylim (yRange);
ylabel('Output power (dBm)','FontSize',10);

%Arrow
text(1570,-19,'\leftarrow');

%Print
width=8;
height=7;
x0=width;
y0=height;
fig.PaperPosition = [x0,y0,width,height];
set(gcf,'units','centimeters','position',[x0,y0,width,height])
res=350 % DPI resolution
print(gcf, '-dpng',['-r' sprintf('%.0f',res)], strcat(dirFigure,nameFigure));



