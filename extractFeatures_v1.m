function features = extractFeatures_v1(ecog, fs)

<<<<<<< HEAD
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

% compute average frequency domain magnitude within bands
% write up says we should be using conv and spectrogram but I didn't
% understand how this worked??

% for i = 1:size(bands, 1)
%     b = bands(i, :);
%     band_diff = b(2) - b(1);
%     %set length of FFT vector in accordance with band - want bands to be same length 
%     % however, bands 2-5 end up being one index longer than band 1 
%     Nfft = length(ecog) * 100 * (10/band_diff); 
%     frequency = ((0:1/Nfft:1-1/Nfft)*fs).'; %define frequency vector 
%     for j = 1:size(ecog, 2)
%         signal = ecog(:,i);
%         f = fft(signal, Nfft); %compute frequency space 
%         mag = abs(f); 
%         % for extracting the magnitudes within the band 
%         band_inds = frequency >= b(1) & frequency <= b(2);
%         avFreq = MovingWinFeats(mag(band_inds), fs, 100e-3, 50e-3, M);
%         disp([num2str(i) ' ' num2str(j) ' ' num2str(length(avFreq))]) %for visualizing progress
%         features(:, end+1) = avFreq(1:5998)'; %chop off extra index 
%     end 
% end 

features = zeros(numWins, size(ecog, 2)*numFeats);

for i = 1:size(ecog, 2) % iterate over number of channels
    signal = ecog(:, i);
    avFreq = MovingWinFeats(signal, fs, winLen, winDisp, M, ...
        @findFreq1, @findFreq2, @findFreq3, @findFreq4, @findFreq5);
    disp([num2str(i) ' ' num2str(length(avFreq))]) %for visualizing progress
    avgFreq = reshape(avFreq, numFeats, numWins)';
    features(:, ((i-1)*numFeats+1).*(1:numFeats)) = avgFreq; 
end
=======
M = @(x) mean(x); % define mean function 
features = zeros(5998, 62);

% compute average time domain voltage for each channel
for i = 1:size(ecog, 2)
    avVoltage = MovingWinFeats(ecog(:,i), fs, 100e-3, 50e-3, M);
    features(:, i) = avVoltage';
end 

% compute average frequency domain magnitude within bands 
% write up says we should be using conv and spectrogram but I didn't
% understand how this worked?? 
bands = [5 15; 20 25; 75 115; 125 160; 160 175];
for i = 1:size(bands, 1)
    b = bands(i, :);
    band_diff = b(2) - b(1);
    %set length of FFT vector in accordance with band - want bands to be same length 
    % however, bands 2-5 end up being one index longer than band 1 
    Nfft = length(ecog) * 100 * (10/band_diff); 
    frequency = ((0:1/Nfft:1-1/Nfft)*fs).'; %define frequency vector 
    for j = 1:size(ecog, 1)
        signal = ecog(i,:);
        f = fft(signal,Nfft); %compute frequency space 
        mag = abs(f); 
        % for extracting the magnitudes within the band 
        band_inds = frequency >= b(1) & frequency <= b(2);
        avFreq = MovingWinFeats(mag(band_inds), fs, 100e-3, 50e-3, M);
        disp([num2str(i) ' ' num2str(j) ' ' num2str(length(avFreq))]) %for visualizing progress
        features(:, end+1) = avFreq(1:5998)'; %chop off extra index 
    end 
end 
>>>>>>> dbc8d7a3ee7022a6ab14622ca687972afe36363c

end 