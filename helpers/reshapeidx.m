function [idxQ, idxIB] = reshapeidx(idx, out)
% split output idx into Q and IB index matrices
idxIB = idx(1:prod(out.IBdim));
idxQ = idx(prod(out.IBdim) + 1:end);
idxIB =reshape(idxIB, out.IBdim);
idxQ = reshape(idxQ, out.Qdim);

end