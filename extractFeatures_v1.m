function features = extractFeatures_v1(ecog, fs)

M = @(x) mean(x); % define mean function
numFeats = 6;     % number of features per channel

winLen = 100e-3;    % window length (s)
winDisp = 50e-3;    % window displacement (s)
numWins = floor((size(ecog, 1)-winLen*fs) / (winDisp*fs)) + 1; % number of windows

bands = [5 15; 20 25; 75 115; 125 160; 160 175];

    function avgfreq = findFreq1(x)
        L = winLen * fs;
        f = fft(x);
        P2 = abs(f/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freq = fs*(0:(L/2))/L;
        
        ind = find((freq >= bands(1, 1)) & (freq <= bands(1, 2)));
        avgfreq = mean(P1(ind));
    end

    function avgfreq = findFreq2(x)
        L = winLen * fs;
        f = fft(x);
        P2 = abs(f/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freq = fs*(0:(L/2))/L;
        
        ind = find((freq >= bands(2, 1)) & (freq <= bands(2, 2)));
        avgfreq = mean(P1(ind));
    end

    function avgfreq = findFreq3(x)
        L = winLen * fs;
        f = fft(x);
        P2 = abs(f/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freq = fs*(0:(L/2))/L;
        
        ind = find((freq >= bands(3, 1)) & (freq <= bands(3, 2)));
        avgfreq = mean(P1(ind));
    end

    function avgfreq = findFreq4(x)
        L = winLen * fs;
        f = fft(x);
        P2 = abs(f/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freq = fs*(0:(L/2))/L;
        
        ind = find((freq >= bands(4, 1)) & (freq <= bands(4, 2)));
        avgfreq = mean(P1(ind));
    end

    function avgfreq = findFreq5(x)
        L = winLen * fs;
        f = fft(x);
        P2 = abs(f/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        freq = fs*(0:(L/2))/L;
        
        ind = find((freq >= bands(5, 1)) & (freq <= bands(5, 2)));
        avgfreq = mean(P1(ind));
    end

features = zeros(numWins, size(ecog, 2)*numFeats);

for i = 1:size(ecog, 2) % iterate over number of channels
    signal = ecog(:, i);
    avFreq = MovingWinFeats(signal, fs, winLen, winDisp, M, ...
        @findFreq1, @findFreq2, @findFreq3, @findFreq4, @findFreq5);
    disp([num2str(i) ' ' num2str(length(avFreq))]) %for visualizing progress
    avgFreq = reshape(avFreq, numFeats, numWins)';
    column = (i-1)*numFeats+1;
    features(:, column:(column+numFeats-1)) = avgFreq; 
end
end 