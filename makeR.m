function R = makeR(features, featsPer)

N = 4;
M = size(features, 1);
numFeats = size(features, 2);
numChannels = numFeats/featsPer;       % assuming 6 features per channel

R = ones(M-N+1, numFeats*N+1);
    %for each row
    for i = 1:(M-N+1) % Edited
        % fixed pretty sure...
        for j = 1:numChannels       % over # channels
            row = features(i:i+N-1, ((j-1)*featsPer+1):((j-1)*featsPer+featsPer))';
            row = reshape(row, 1, featsPer*N);
            idx = (j-1)*N*featsPer+2;
            R(i, idx:idx+N*featsPer-1) = row;

%             row = features(i:i+N-1, :)';
%             row = reshape(row, 1, featsPer*N*numChannels);
%             R(i, 2:end) = row;
        end
    end 
    R(end+1:end+N-1, :) = R(1:N-1, 1:numFeats*N+1);

end