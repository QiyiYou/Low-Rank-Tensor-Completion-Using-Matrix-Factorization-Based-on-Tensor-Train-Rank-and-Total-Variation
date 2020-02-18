clc;
clear; close all;
addpath(genpath(cd));
rand('seed',213412); 

Nway = [4 4 4 4 4 4 4 4 3];     % 9th-order dimensions for KA
N = numel(Nway);                   
I1 = 2; J1 = 2;                 % KA parameters
SR = 0.1;                       % Sample ratio (SR), e.g. 0.1 = 10% known samples
X0 = double(imread('lena.bmp'));

%% Generate known data
P = round(SR*prod(Nway));  
Known = randsample(prod(Nway),P);
[Known,~] = sort(Known);
%% Missing data
X = CastImageAsKet22( X0, Nway, I1, J1);    
Xkn          = X(Known);
Xmiss        = zeros(Nway);
Xmiss(Known) = Xkn;
Xmiss = CastKet2Image22(Xmiss,256,256,I1,J1);

%% use TT-TV
opts=[];
opts.alpha  = weightTC(Nway);
opts.tol    = 1e-4;
opts.maxit  = 200;
opts.X      = X;    
opts.rho    = 10^(-3);
opts.th     = 0.01;
opts.lambda = 0.3; 
opts.beta1  = 5*10^(-3); 
opts.beta2  = 0.1; 
opts.beta3  = 0.3; 

[X_TT_TV, Out_TT_TV] = TT_TV( Xkn, Known, Nway, opts );

X_TT_TV = CastKet2Image22(X_TT_TV,256,256,I1,J1);
X_TT_TV = min( 255, max( X_TT_TV, 0 ));
    
PSNRvector=zeros(1,3);
for i=1:1:3
    PSNRvector(i)=psnr(X0(:,:,i),X_TT_TV(:,:,i));
end
PSNR_TT_TV = mean(PSNRvector);
        
SSIMvector=zeros(1,3);
for i=1:1:3
    SSIMvector(i)=ssim(X0(:,:,i),X_TT_TV(:,:,i));
end
SSIM_TT_TV = mean(SSIMvector);

figure,imshow(uint8(X_TT_TV))
display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f,lambda=%.1f,beta1=%.3f,beta2=%.1f,beta3=%.1f',PSNR_TT_TV, SSIM_TT_TV, opts.th, opts.lambda, opts.beta1, opts.beta1, opts.beta3))     