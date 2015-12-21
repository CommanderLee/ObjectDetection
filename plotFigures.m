% plot figures
%% cross loss
clear all;
close all;
xLabel = {'Car vs All', 'Pedestrian vs All', 'Car vs Pedestrian'};
x = 1:3;
y1 = [8.75, 4.07, 3.48] / 100;
y2 = [6.65, 4.13, 2.75] / 100;

hold on;
plot(x, y1, 'ro-', 'LineWidth', 3);
plot(x, y2, 'b*--', 'LineWidth', 3);

legend('Before Multiplication', 'After Multiplication', 'Location', 'northeast');

strValuesA = strtrim(cellstr(num2str([y1(:) * 100],'%.2f')));
text(x, y1, strValuesA, 'Color', 'red', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
strValuesP = strtrim(cellstr(num2str([y2(:) * 100],'%.2f')));
text(x, y2, strValuesP, 'Color', 'blue', 'FontSize', 12, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center');

set(gca, 'xtick', 1:3, 'xticklabel', xLabel);
% Ref: http://www.mathworks.com/matlabcentral/answers/94708-how-do-i-change-my-y-axis-or-x-axis-values-to-percentage-units-and-have-these-changes-reflected-on
% Convert y-axis values to percentage values by multiplication
a=[cellstr(num2str(get(gca,'ytick')'*100))]; 
% Create a vector of '%' signs
pct = char(ones(size(a,1),1)*'%'); 
% Append the '%' signs after the percentage values
new_yticks = [char(a),pct];
% 'Reflect the changes on the plot
set(gca,'yticklabel',new_yticks);

title('Cross Loss');

%% accuracy, precision, recall
clear all;
close all;
xLabel = {'Car vs All', 'Pedestrian vs All', 'Car vs Pedestrian'};
x = 1:3;
% A, P, and R
a1 = [89.46, 95.47, 96.08] / 100;
p1 = [87.41, 80.44, 97.46] / 100;
r1 = [88.32, 79.52, 97.59] / 100;

a2 = [93.60, 95.89, 97.32] / 100;
p2 = [92.96, 98.26, 98.26] / 100;
r2 = [88.72, 96.56, 96.56] / 100;

subplot(1, 2, 1);
hold on;
axis([1, 3, 0.78, 1.00]);

plot(x, a1, 'ro-', 'LineWidth', 3);
plot(x, p1, 'b*--', 'LineWidth', 3);
plot(x, r1, 'gx:', 'LineWidth', 3);

legend('Accuracy', 'Precision', 'Recall', 'Location', 'southeast');

strValuesA = strtrim(cellstr(num2str([a1(:) * 100],'%.2f')));
text(x, a1, strValuesA, 'Color', 'red', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
strValuesP = strtrim(cellstr(num2str([p1(:) * 100],'%.2f')));
text(x, p1, strValuesP, 'Color', 'blue', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
strValuesR = strtrim(cellstr(num2str([r1(:) * 100],'%.2f')));
text(x, r1, strValuesR, 'Color', 'green', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

set(gca, 'xtick', 1:3, 'xticklabel', xLabel);
% Ref: http://www.mathworks.com/matlabcentral/answers/94708-how-do-i-change-my-y-axis-or-x-axis-values-to-percentage-units-and-have-these-changes-reflected-on
% Convert y-axis values to percentage values by multiplication
a=[cellstr(num2str(get(gca,'ytick')'*100))]; 
% Create a vector of '%' signs
pct = char(ones(size(a,1),1)*'%'); 
% Append the '%' signs after the percentage values
new_yticks = [char(a),pct];
% 'Reflect the changes on the plot
set(gca,'yticklabel',new_yticks);
title('Before multiplication');

subplot(1, 2, 2);
hold on;
axis([1, 3, 0.78, 1.00]);

plot(x, a2, 'ro-', 'LineWidth', 3);
plot(x, p2, 'b*--', 'LineWidth', 3);
plot(x, r2, 'gx:', 'LineWidth', 3);

legend('Accuracy', 'Precision', 'Recall', 'Location', 'southeast');

strValuesA = strtrim(cellstr(num2str([a2(:) * 100],'%.2f')));
text(x, a2, strValuesA, 'Color', 'red', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
strValuesP = strtrim(cellstr(num2str([p2(:) * 100],'%.2f')));
text(x, p2, strValuesP, 'Color', 'blue', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
strValuesR = strtrim(cellstr(num2str([r2(:) * 100],'%.2f')));
text(x, r2, strValuesR, 'Color', 'green', 'FontSize', 12, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

set(gca, 'xtick', 1:3, 'xticklabel', xLabel);
% Ref: http://www.mathworks.com/matlabcentral/answers/94708-how-do-i-change-my-y-axis-or-x-axis-values-to-percentage-units-and-have-these-changes-reflected-on
% Convert y-axis values to percentage values by multiplication
a=[cellstr(num2str(get(gca,'ytick')'*100))]; 
% Create a vector of '%' signs
pct = char(ones(size(a,1),1)*'%'); 
% Append the '%' signs after the percentage values
new_yticks = [char(a),pct];
% 'Reflect the changes on the plot
set(gca,'yticklabel',new_yticks);
title('After multiplication');