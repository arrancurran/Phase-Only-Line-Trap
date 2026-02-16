% Expanded version of otslm_line_trap.m with inlined OTSLM logic
% This script reproduces the behaviour of otslm_line_trap.m but
% implements the required OTSLM functions locally so everything
% happens in one file.

%% Simulation parameters
radius = 10;          % Inverse length of the line trap (sinc radius)
theta = 0;             % Rotation of pattern (degrees)

sz = [512, 512];       % Size of pattern
o = 400;               % Region of interest size in output
padding = 500;         % Padding for FFT

incident = ones(sz);   % Incident beam (use uniform illumination)

% Functions used for generating figures
zoom = @(im) im(round(size(im, 1)/2)+(-o:o), round(size(im, 2)/2)+(-o:o));
visualize = @(pattern) zoom(abs(local_visualise_fft(pattern, ...
    'padding', padding, 'incident', incident)).^2);
  
%% Roichman and Grier (2006) using 1-D line trap with amplitude encoded in 2D

sinc_pattern = local_sinc(sz, radius, 'type', '1d', 'angle_deg', theta);
[pattern, assigned] = local_encode1d(sinc_pattern, ...
    'angle_deg', theta, 'scale', 200);
  
% Apply a checkerboard-like linear pattern to assigned regions
checker_line = local_linear(sz, 10, 'angle_deg', 90);
checker_spot = local_linear(sz, 3, 'angle_deg', 90); %#ok<NASGU> kept for reference
pattern(assigned) = pattern(assigned) + checker_line(assigned); 
pattern(~assigned) = checker_spot(~assigned);

pattern = local_finalize(pattern);

subplot(2, 1, 1);
imagesc(pattern);

subplot(2, 1, 2);
imagesc(visualize(pattern));

%% Change properties of all figures

for ii = 1:2
  subplot(2, 1, ii);
  axis('image');
  colormap('gray');
  set(gca,'YTickLabel', [], 'XTickLabels', []);
end


%% ---- Local helpers implementing required parts of +otslm -----------------

function pattern = local_sinc(sz, radius, varargin)
% Based on otslm.simple.sinc
% Generates a sinc pattern for 1D line trap.

p = inputParser;
p = local_addGridParameters(p, sz);
p.parse(varargin{:});

% Calculate radial coordinates
gridParameters = local_expandGridParameters(p);
[xx, yy, rr] = local_grid(sz, gridParameters{:}); %#ok<ASGLU>

% Generate pattern (use MATLAB sinc definition sinc(x) = sin(pi x)/(pi x))
if strcmpi(p.Results.type, '1d')
  x = xx ./ radius(1);
  pattern = local_builtin_sinc(x);
elseif strcmpi(p.Results.type, '2d')
  r = rr ./ radius(1);
  pattern = local_builtin_sinc(r);
elseif strcmpi(p.Results.type, '2dcart')
  if numel(radius) == 2
    pattern = local_builtin_sinc(xx./radius(1)) .* ...
              local_builtin_sinc(yy./radius(2));
  else
    pattern = local_builtin_sinc(xx./radius(1)) .* ...
              local_builtin_sinc(yy./radius(1));
  end
else
  error('Unknown value passed for type argument');
end

% Normalize pattern to max value of 1
pattern = pattern ./ max(abs(pattern(:)));
end

function pattern = local_linear(sz, spacing, varargin)
% Based on otslm.simple.linear
% Generates a linear gradient used as a grating/checker pattern.

p = inputParser;
p = local_addGridParameters(p, sz, 'skip', 'type', 'centre', [1, 1]);
p.parse(varargin{:});

% Generate grid of points
gridParameters = local_expandGridParameters(p);
[xx, yy] = local_grid(sz, gridParameters{:}); %#ok<ASGLU>

% Check for valid spacing
if any(spacing == 0)
  warning('Spacing should be non-zero, using spacing of Inf');
  spacing(spacing == 0) = Inf;
end

% Generate pattern
if numel(spacing) == 1
  pattern = xx ./ spacing;
elseif numel(spacing) == 2
  pattern = xx ./ spacing(1) + yy ./ spacing(2);
else
  error('Spacing must be 1 or 2 elements');
end
end

function [xx, yy, rr, phi] = local_grid(sz, varargin)
% Based on otslm.simple.grid
% Generates a grid of points centred in the image.

p = inputParser;
p = local_addGridParameters(p, sz);
p.parse(varargin{:});

% Generate grid
if p.Results.gpuArray
  [xx, yy] = meshgrid(gpuArray(1:sz(2)), gpuArray(1:sz(1)));
else
  [xx, yy] = meshgrid(1:sz(2), 1:sz(1));
end

% Move centre of pattern
xx = xx - p.Results.centre(1);
yy = yy - p.Results.centre(2);

% Apply rotation to pattern
angle = local_getParameterAngle(p, 0.0);
xxr = cos(angle).*xx - sin(angle).*yy;
yyr = sin(angle).*xx + cos(angle).*yy;
xx = xxr;
yy = yyr;

% Apply aspect ratio
yy = yy * p.Results.aspect;

% Apply offset in transformed coordinates
xx = xx - p.Results.offset(1);
yy = yy - p.Results.offset(2);

if nargout > 2
  % Calculate r
  if strcmpi(p.Results.type, '1d')
    rr = sqrt(xx.^2);
  elseif strcmpi(p.Results.type, '2d')
    rr = sqrt(xx.^2 + yy.^2);
  else
    error('Unknown type, must be 1d or 2d');
  end

  if nargout > 3
    % Calculate phi
    phi = atan2(yy, xx);
  end
end
end

function p = local_addGridParameters(p, sz, varargin)
% Based on otslm.simple.private.addGridParameters

assert(numel(sz) == 2, 'sz must be 2 element vector');

parameters = {'centre', 'offset', 'type', 'aspect', 'angle', ...
  'angle_deg', 'gpuArray'};

% Allow user to change input parser defaults
ip = inputParser;
ip.addParameter('centre', [ sz(2)/2, sz(1)/2 ]);
ip.addParameter('offset', [0, 0]);
ip.addParameter('type', '2d');
ip.addParameter('aspect', 1.0);
ip.addParameter('angle', []);
ip.addParameter('angle_deg', []);
ip.addParameter('gpuArray', false);
ip.addParameter('skip', {});
ip.parse(varargin{:});

% Add parameters to result input parser
for ii = 1:length(parameters)
  if ~any(strcmpi(parameters{ii}, ip.Results.skip))
    p.addParameter(parameters{ii}, ip.Results.(parameters{ii}));
  end
end
end

function params = local_expandGridParameters(p)
% Based on otslm.simple.private.expandGridParameters

parameters = {'centre', 'offset', 'type', 'aspect', 'angle', ...
  'angle_deg', 'gpuArray'};

num = 0;
params = {};
for ii = 1:length(parameters)
  if isfield(p.Results, parameters{ii})
    params{2*num+1} = parameters{ii}; %#ok<AGROW>
    value = p.Results.(parameters{ii});
    % Type could be longer than 2 chars (such as 2dcart)
    if strcmpi(parameters{ii}, 'type') && numel(value) > 2
      value = value(1:2);
    end
    params{2*num+2} = value; %#ok<AGROW>
    num = num + 1;
  end
end
end

function [rad, deg] = local_getParameterAngle(p, defaultAngle)
% Based on otslm.simple.private.getParameterAngle
% Get angle from inputParser results in radians or degrees

rad = [];
if ~isempty(p.Results.angle)
  rad = p.Results.angle;
end
if ~isempty(p.Results.angle_deg)
  assert(isempty(rad), 'Angle set multiple times');
  rad = p.Results.angle_deg * pi/180.0;
end
if isempty(rad)
  if nargin == 1
    rad = 0.0;
  else
    rad = defaultAngle;
  end
end

if nargout > 1
  deg = 180/pi * rad; %#ok<NASGU>
end
end

function [pattern, assigned] = local_encode1d(target, varargin)
% Based on otslm.tools.encode1d
% Encode the target 1-D amplitude pattern into a phase-only mask.

p = inputParser;
p.addParameter('scale', 1.0);
p.addParameter('angle', []);
p.addParameter('angle_deg', []);
p.parse(varargin{:});

% Parse the angle
angle = [];
if ~isempty(p.Results.angle)
  assert(isempty(angle), 'Angle set multiple times');
  angle = p.Results.angle;
end
if ~isempty(p.Results.angle_deg)
  assert(isempty(angle), 'Angle set multiple times');
  angle = p.Results.angle_deg * pi/180.0;
end
if isempty(angle)
  angle = 0.0;
end

% Generate grid
[~, yy] = local_grid(size(target), 'angle', angle);

% Generate the pattern
phi = (target >= 0)*0.5;
assigned = (abs(yy) < abs(target*p.Results.scale));
pattern = phi .* assigned;
end

function U = local_make_beam(phase, varargin)
% Based on otslm.tools.make_beam
% Combine phase, amplitude and incident patterns into complex field.

p = inputParser;
p.addParameter('amplitude', []);
p.addParameter('incident', []);
p.parse(varargin{:});

amplitude = p.Results.amplitude;
incident = p.Results.incident;

% Handle default value for amplitude
if isempty(amplitude) && ~isempty(phase)
  amplitude = ones(size(phase));
end

% Handle default value for phase
if isempty(phase)
  if ~isempty(amplitude)
    phase = zeros(size(amplitude));
  elseif ~isempty(incident)
    phase = zeros(size(incident));
    amplitude = ones(size(incident));
  else
    error('Must have at least one input image');
  end
end

% Handle default value for incident
if isempty(incident)
  psz = size(phase);
  incident = ones(psz(1:2));
end

% Ensure incident and amplitude are volumes if phase is a volume
if size(phase, 3) ~= size(incident, 3) && size(incident, 3) == 1
  incident = repmat(incident, [1, 1, size(phase, 3)]);
end
if size(phase, 3) ~= size(amplitude, 3) && size(amplitude, 3) == 1
  amplitude = repmat(amplitude, [1, 1, size(phase, 3)]);
end

% Check sizes of input images
assert(all(size(incident) == size(phase)), ...
  'Incident size must match phase size');
assert(all(size(amplitude) == size(phase)), ...
  'Amplitude size must match phase size');

% Allow the user to pass in a single complex amplitude or
% separate phase and amplitude matrices
if isreal(phase)
  % Generate combined pattern
  U = amplitude .* exp(1i*phase) .* incident;
else
  % The input is a complex amplitude
  U = phase .* incident;
end
end

function output = local_visualise_fft(phase, varargin)
% Simplified version of otslm.tools.visualise for method='fft', type='farfield'.

p = inputParser;
p.addParameter('method', 'fft');
p.addParameter('type', 'farfield');
p.addParameter('amplitude', []);
p.addParameter('incident', []);
p.addParameter('z', 0.0); %#ok<NASGU> kept for reference
p.addParameter('padding', ceil(size(phase)/2));
p.addParameter('trim_padding', false);
p.addParameter('NA', 0.1); %#ok<NASGU> kept for reference
p.addParameter('resample', []);
p.parse(varargin{:});

assert(strcmpi(p.Results.method, 'fft'), 'local_visualise_fft only supports method="fft"');
assert(strcmpi(p.Results.type, 'farfield'), 'local_visualise_fft only supports type="farfield"');

% Create a complex beam from the inputs
U = local_make_beam(phase, ...
    'incident', p.Results.incident, ...
    'amplitude', p.Results.amplitude);

% (Optional) resampling block omitted here because it is not used

% Apply forward FFT propagation with padding
output = local_fft_forward_simple(U, ...
    'padding', p.Results.padding, ...
    'trim_padding', p.Results.trim_padding);
end

function [output] = local_fft_forward_simple(pattern, varargin)
% Functional equivalent of otslm.tools.prop.FftForward.simple
% for z = 0 (no additional lens phase).

p = inputParser;
p.addParameter('axial_offset', 0.0); %#ok<NASGU> kept for reference
p.addParameter('NA', 0.1);          %#ok<NASGU> kept for reference
p.addParameter('padding', ceil(size(pattern)/2));
p.addParameter('trim_padding', true);
p.parse(varargin{:});

padIn = p.Results.padding;

% Parse padding as in FftBase
switch numel(padIn)
  case 0
    padding = [0, 0];
  case 1
    padding = [1, 1].*padIn;
  case 2
    padding = padIn;
  otherwise
    error('Padding must be 0, 1 or 2 element vector');
end

sz = size(pattern);

% Total size including padding
total_sz = sz + 2*padding;

% Allocate array and insert input into central region
data = zeros(total_sz, 'like', pattern);
py = padding(1);
px = padding(2);
data(py+1:py+sz(1), px+1:px+sz(2)) = pattern;

% Forward 2-D FFT with shift and normalization
output = fftshift(fft2(data)) ./ numel(data);

% Remove padding if requested
if p.Results.trim_padding
  output = output(py+1:py+sz(1), px+1:px+sz(2));
end
end

function pattern = local_finalize(pattern, varargin)
% Based on otslm.tools.finalize for phase-only SLM, no amplitude.

p = inputParser;
p.addParameter('modulo', []);
p.addParameter('device', 'slm');
p.addParameter('colormap', []);
p.addParameter('rpack', []);
p.addParameter('amplitude', []);
p.addParameter('encodemethod', 'checker'); %#ok<NASGU> kept for reference
p.parse(varargin{:});

% Set default colour map
cmap = p.Results.colormap;
if isempty(cmap)
  if strcmpi(p.Results.device, 'slm')
    cmap = 'pmpi';
  elseif strcmpi(p.Results.device, 'dmd')
    cmap = 'gray';
  else
    error('Unknown device');
  end
end

% Set default rpack
rpack = p.Results.rpack;
if isempty(rpack)
  if strcmpi(p.Results.device, 'slm')
    rpack = 'none';
  elseif strcmpi(p.Results.device, 'dmd')
    rpack = '45deg';
  else
    error('Unknown device');
  end
end

% Handle default modulo value for pattern
modv = p.Results.modulo;
if isempty(modv)
  if strcmpi(p.Results.device, 'slm')
    modv = 1.0;
  elseif strcmpi(p.Results.device, 'dmd')
    modv = 'none';
  else
    error('Unknown device');
  end
end

% (Amplitude encoding branch omitted; not used in this script)

% Apply modulo to pattern
if ischar(modv) && strcmpi(modv, 'none')
  % Nothing to do
elseif ~ischar(modv)
  pattern = mod(pattern, modv);
else
  error('Unknown modulo argument value');
end

% Apply colour map
pattern = local_colormap(pattern, cmap);

% Apply rotation to pattern
switch rpack
  case 'none'
    % Nothing to do
  case '45deg'
    sz = size(pattern);
    npattern = zeros(ceil(sz(2)/2) + sz(1) - 1, ...
        ceil((sz(2)+1)/2) + sz(1) - 1);

    [ox, oy] = meshgrid(1:sz(2), 1:sz(1));
    nx = ceil((ox+1)/2) + oy - 1;
    ny = ceil(ox/2) + (sz(1) - 1) - (oy - 1);
    ind = sub2ind(size(npattern), ny, nx);

    npattern(ind) = pattern;
    pattern = cast(npattern, 'like', pattern);
  otherwise
    error('Unknown option for rpack');
end
end

function pattern = local_colormap(pattern, cmap, varargin)
% Based on otslm.tools.colormap, restricted to string cmaps.

p = inputParser;
p.addParameter('inverse', false);
p.parse(varargin{:});

% Check that we actually have a colormap
if isempty(cmap)
  return;  % Nothing to do
end

if ischar(cmap)
  if ~p.Results.inverse
    switch cmap
      case 'pmpi'
        pattern = pattern*2*pi - pi;
      case '2pi'
        pattern = pattern*2*pi;
      case 'bin'
        error('Binary colormap not used in this script');
      case 'gray'
        % Nothing to do
      otherwise
        error('Unrecognized colormap string');
    end
  else
    switch cmap
      case 'pmpi'
        pattern = (pattern + pi)./(2*pi);
      case '2pi'
        pattern = pattern./(2*pi);
      case 'bin'
        error('Inverse binary colormap not used');
      case 'gray'
        % Nothing to do
      otherwise
        error('Unrecognized colormap string');
    end
  end
else
  error('local_colormap in this file only supports string cmaps');
end
end

function y = local_builtin_sinc(x)
% Helper: MATLAB-like sinc(x) = sin(pi*x)/(pi*x), with x=0 -> 1.
y = ones(size(x), 'like', x);
nonzero = (x ~= 0);
y(nonzero) = sin(pi*x(nonzero)) ./ (pi*x(nonzero));
end
