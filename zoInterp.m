function y = zoInterp(x, numInterp)

y = reshape(repmat(x,numInterp,1),size(x,1),size(x, 2)*numInterp);

end

