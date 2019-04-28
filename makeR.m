function R = makeR(features)

N = 4;
M = size(features, 1);
numFeats = size(features, 2);
featsPer = 6;                   % features/channel
numChannels = numFeats/featsPer;       % assuming 6 features per channel

R = ones(M-N, numFeats*N+1);
%for each row
for i = 1:(M-N+1)
    % fixed pretty sure...
    for j = 1:numChannels       % over # channels
        row = features(i:i+N-1, ((j-1)*featsPer+1):((j-1)*featsPer+featsPer))';
        row = reshape(row, 1, featsPer*N);
        idx = (j-1)*N*featsPer+2;
        R(i, idx:idx+N*featsPer-1) = row;
    end
end

end