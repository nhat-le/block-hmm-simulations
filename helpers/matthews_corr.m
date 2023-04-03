function mcc = matthews_corr(cm)
% calculating the matthews correlation coefficient from a confusion matrix
% input

% tp = conf_matrix(1,1);
% tn = conf_matrix(2,2);
% fp = conf_matrix(2,1);
% fn = conf_matrix(1,2);

% mcc = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

% Multiclass version from http://arxiv.org/pdf/1008.2908v1.pdf
N = size(cm, 1);
num = 0;
d1 = 0;
d2 = 0;
for k = 1:N
	clk = 0;
	cgf = 0;
	ckl = 0;
	cfg = 0;

	for l = 1:N
		for m = 1:N
			num = num + cm(k,k)*cm(m,l) - cm(l,k)*cm(k,m);
		end

		clk = clk + cm(l, k);
		ckl = ckl + cm(k, l);

		if (l ~= k)
			for g = 1:N
				cgf = cgf + cm(g, l);
				cfg = cfg + cm(l, g);
			end
		end
	end

	d1 = d1 + clk*cgf;
	d2 = d2 + ckl*cfg;
end

mcc = num/(sqrt(d1)*sqrt(d2));

if ~(mcc >= -1 && mcc <= 1)
	fprintf(1, 'WARNING: MCC should be in [-1, 1], is %f %s\n', mcc, mat2str(cm));
end

end