function [ path ] = OpenFile()
%OPENFILE Summary of this function goes here
%   Detailed explanation goes here
    [FileName,PathName] = uigetfile('*.*');
    path = [PathName, FileName];
end

