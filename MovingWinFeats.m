function features = MovingWinFeats(x, fs, winLen, winDisp, featFn, varargin)
% MovingWinFeats calculates features on a signal
% inputs:   x (signal vector)
%           fs (sampling frequency Hz)
%           winLen (length of sliding window - s)
%           winDisp (length of displacement - s)
%           featFn (function for feature calculation)
%           optional parameters are for additional features
% outputs:  features (vector of feature values)

numWins = floor((length(x)-winLen*fs) / (winDisp*fs)) + 1; % number of windows
nVarargs = length(varargin);        % number of extra features
disp(nVarargs)
features = zeros(1, numWins*(nVarargs+1));

warning('off','all')

% calculate features within window
for i = 0:numWins-1
    index = i*winDisp*fs+1;
    features(i*(nVarargs+1)+1) = featFn(x(index:(index+winLen*fs-1)));
    if (nVarargs ~= 0)
        for j = 1:nVarargs  % iterate over extra feature functions
            features(i*(nVarargs+1)+1+j) = varargin{j}(x(index:(index+winLen*fs-1)));
        end
    end
end

end