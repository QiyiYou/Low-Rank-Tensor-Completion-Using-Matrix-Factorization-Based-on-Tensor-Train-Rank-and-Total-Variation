function [M, Out_TT_TV] = TT_TV( data, known, Nway, opts )

    Orig = opts.X;
    alpha = opts.alpha; th = opts.th; maxiter = opts.maxit; tol = opts.tol;
    lambda = opts.lambda; beta1 = opts.beta1; beta2 = opts.beta2; beta3 = opts.beta3; rho = opts.rho;
    %% Initialization
    R = 256; C = 256; I1 = 2; J1 = 2;
    N = length(Nway);
    M = initialization_M(Nway,known,data); 
    [~,ranktube] = SVD_MPS_Rank_Estimation(Orig,th);   % Initial TT ranks
    [X,Y] = initialMatrix(N,Nway,ranktube);
     
    dimL = zeros(1,N-1);
    dimR = zeros(1,N-1);
    IL = 1;
    for k = 1:N-1
        dimL(k) = IL*Nway(k);
        dimR(k) = prod(Nway)/dimL(k);
        IL = dimL(k);
    end 
    
    X0 = X; Y0 = Y;  M0 = M;  
    X = cell(1,N-1); Y = cell(1,N-1);
    
    N = length(Nway);
    k = 1;
    relerr = [];
    relerr(1) = 1;
   
    %% Start Time measure
    t0=tic;
    while relerr(k) > tol
        k = k+1;
        Mlast = M;
        %% update (X,Y)
        for n = 1:N-1
            M_Temp = reshape(M0,[dimL(n) dimR(n)]);
            X{n}   = (alpha(n)*M_Temp*Y0{n}' + rho*X0{n})*pinv( alpha(n)*Y0{n}*Y0{n}' + rho*eye(size(Y0{n}*Y0{n}')));
            Y{n}   = pinv(alpha(n)*X{n}'*X{n} + rho*eye(size(X{n}'*X{n})))*(alpha(n)*X{n}'*M_Temp +rho*Y0{n});
        end
        
        %% update M by ADMM
        [M] = update_M(data,known,Nway,N,X,Y,M0,alpha,lambda,beta1,beta2,beta3,rho);
        
        % Calculate relative error
        relerr(k) = abs(norm(M(:)-Mlast(:)) / norm(Mlast(:)));

        Img  = CastKet2Image22(M,R,C,I1,J1);
        Img0 = CastKet2Image22(Orig,R,C,I1,J1);
        if k == 2 || mod(k-1,20) ==0 
           fprintf('  %d:  %f \n',k-1,RSE(Img(:),Img0(:)));
        end

        X0=X; Y0=Y; M0=M;
        %% check stopping criterion
        if k > maxiter || (relerr(k)-relerr(k-1) > 0)
            break
        end
    end
    %% Stop Time measure
    time = toc(t0);
    Out_TT_TV.time = time;
    Out_TT_TV.relerr = relerr;
end
function [X0,Y0] = initialMatrix(N,Nway,ranktube)
    X0 = cell(1,N-1);Y0 = cell(1,N-1);
    dimL = zeros(1,N-1);
    dimR = zeros(1,N-1);
    IL = 1;
    for k = 1:N-1
        dimL(k) = IL*Nway(k);
        dimR(k) = prod(Nway)/dimL(k);
        %
        X0{k} = randn(dimL(k),ranktube(k));
        Y0{k} = randn(ranktube(k),dimR(k));
        %uniform distribution on the unit sphere
        X0{k} = bsxfun(@rdivide,X0{k},sqrt(sum(X0{k}.^2,1)));
        Y0{k} = bsxfun(@rdivide,Y0{k},sqrt(sum(Y0{k}.^2,2)));
        %
        IL = dimL(k);
    end   
end