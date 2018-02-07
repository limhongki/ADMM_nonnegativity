function x = bin_read(filename)

fid = fopen(filename, 'r');
n = 128;
m = 128; 
bin = single(fread(fid,'float'));
l = length(bin)/m/n;
x = reshape(bin, [n m l]);
fclose(fid);
x = x(:,:,end:-1:1);
end

