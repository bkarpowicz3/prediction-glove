function Y = linreg(features, labels)

N = 3;
M = size(features, 1);
numFeats = size(features, 2);
featsPer = 6;                   % features/channel
numChannels = numFeats/featsPer;       % assuming 6 features per channel

R = ones(M-N, numFeats*N+1);
%for each row
for i = 1:(M-N+1)
%     row = zeros(numFeats*N);
%     row(end+1) = 1;
%     %for each feature -- should this be done on features transpose? so that
%     %it goes over time? not sure how to adapt here
%     for j = 1:size(features, 2)
%         row(end+1:end+N) = features(start:start+N-1, j);
%     end 
%     start = start + 1;
%     R(i, :) = row;
    
    % fixed pretty sure...
    for j = 1:numChannels       % over # channels
        row = features(i:i+N-1, ((j-1)*featsPer+1):((j-1)*featsPer+featsPer))';
        row = reshape(row, 1, featsPer*N);
        idx = (j-1)*N*featsPer+2;
        R(i, idx:idx+N*featsPer-1) = row;
    end
end 

a = R'*R;
ainv = a \ eye(size(a, 1)); % need to compute inverse this way or else you get Inf 
B = ainv*(R'*labels(N:end, :));
Y = R*B;

end

