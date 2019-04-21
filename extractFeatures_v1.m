function features = extractFeatures_v1(ecog, fs)

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

end 