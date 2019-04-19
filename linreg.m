function Y = linreg(features, labels)

R = [];
start = 1;
N = 3;
M = 2950;

%for each row
for i = 1:M
    row = [];
    row(end+1) = 1;
    %for each feature -- should this be done on features transpose? so that
    %it goes over time? not sure how to adapt here
    for j = 1:size(features,2)
        row(end+1:end+N) = features(start:start+N-1, j);
    end 
    start = start + 1;
    R = [R; row];
end 

a = R'*R;
ainv = a / eye(size(a)); % need to compute inverse this way or else you get Inf 
B = ainv*R'*labels;
Y = R*B;

end

