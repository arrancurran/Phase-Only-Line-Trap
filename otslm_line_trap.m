% Simulation of 1D line trap Based on Yoels paper

% Add toolbox to path
addpath('../');

radius = 10;          % Inverse length of the line trap (sinc radius)
theta = 0;         % Rotation of pattern (degrees)

sz = [512, 512];      % Size of pattern
o = 400;               % Region of interest size in output
padding = 500;        % Padding for FFT

% incident = [];        % Incident beam (use default in visualize)
incident = ones(sz);  % Incident beam (use uniform illumination)

% Functions used for generating figures
zoom = @(im) im(round(size(im, 1)/2)+(-o:o), round(size(im, 2)/2)+(-o:o));
visualize = @(pattern) zoom(abs(otslm.tools.visualise(pattern, ...
    'method', 'fft', 'padding', padding, 'incident', incident)).^2);
  
%% Roichman and Grier (2006) using 1-D line trap with amplitude encoded in 2d

sinc = otslm.simple.sinc(sz, radius, 'type', '1d', 'angle_deg', theta);
[pattern, assigned] = otslm.tools.encode1d(sinc, ...
    'angle_deg', theta, 'scale', 200);
  
% Apply a checkerboard to unassigned regions
checker_line = otslm.simple.linear(sz, 10, 'angle_deg', 90);
checker_spot = otslm.simple.linear(sz, 6, 'angle_deg', 90);
pattern(assigned) = pattern(assigned) + checker_line(assigned); 
pattern(~assigned) = checker_spot(~assigned);

pattern = otslm.tools.finalize(pattern);

subplot(2, 1, 1);
imagesc(pattern);

subplot(2, 1, 2);
imagesc(log(visualize(pattern)));

%% Change properties of all figures

for ii = 1:2
  subplot(2, 1, ii);
  axis('image');
  colormap('gray');
  
  set(gca,'YTickLabel', [], 'XTickLabels', []);
end
