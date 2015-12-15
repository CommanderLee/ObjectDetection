function ratio = rectOverlap(top1, left1, bot1, right1, top2, left2, bot2, right2)
% Calculate the overlap ratio of two rectangles

s1 = (bot1 - top1) * (right1 - left1);
s2 = (bot2 - top2) * (right2 - left2);
si = max(0, min(bot1, bot2) - max(top1, top2)) * max(0, min(right1, right2) - max(left1, left2));
ratio = si / (s1 + s2 - si);
end