function [fit] = func(x, fun)
%benchmark functions

% Developed by: Dr. Mahamed G.H. Omran (omran.m@gust.edu.kw) 12-May-2011

% x is a D-dimensional vector representing the solution we want to evalaute
% fun is the indix of the function of interest
% fit is the fitness of x (i.e. fit = f(x))

D = length(x);

switch(fun)
    case {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}, % CEC'05 functions
        % NOTE:
        % For these functions to work, you need to include the following
        % lines in your file:
        % global  initial_flag
        % initial_flag = 0;
        % In addition, vector x should be x(1,D) where D is the problem's
        % dimension
        fit = benchmark_func(x, fun);
    case {26}, % Sphere function
        fit = sum(x.^2);
    case {27}, % Rastrigin
        fit = 10*D + sum(x.^2 - 10*cos(2.*pi.*x));
    case {28}, %Six Hump Camel bsck
        fit = 4*x(1).^2 - 2.1*x(1).^4 + (1/3)*x(1).^6 + x(1)*x(2) - 4*x(2)^2 + 4*x(2).^4;
    case {29}, %Step
        fit = sum(floor(x + 0.5).^2);
    case {30}, %Rosenbrock
        f = 0;
        for i = 1:1:D-1
            f = f + 100*((x(i+1) - x(i)^2)^2) + (1 - x(i))^2;
        end
        fit = f;
    case {31}, %Ackley
       fit = -20*exp(-0.2*sqrt(sum(x.^2)/D)) - exp(sum(cos(2*pi .*x))/D) + 20 + exp(1);
    case {32}, %Griewank
        sum1 = 0;
        prod1 = 1;
        for i = 1:1:D
            sum1 = sum1 + (x(i)^2/4000);
            prod1 = prod1*cos(x(i)/sqrt(i));
        end
        fit = sum1 - prod1 + 1;
    case {33}, %Salomon
        fit = -cos(2*pi*sqrt(sum(x.^2))) + 0.1*sqrt(sum(x.^2)) + 1;
    case {34}, %Normalized Schwefel
        fit = sum(-x.*sin(sqrt(abs(x))))/D;
    case {35}, %Quartic function
        sum1 = 0;
        for i = 1:1:D
            sum1 = sum1 + i*x(i)^4;
        end
        fit = sum1 + rand();
    case {36}, %Rotated hyper-ellipsoid
        sum2 = 0;
        for i = 1:1:D
            sum1 = 0;
            for j = 1:1:i
                sum1 = sum1 + x(j);
            end
            sum2 = sum2 + sum1^2;
        end
        fit = sum2;
    case {37}, %Norwegian function
        prod1 = 1;
        for i=1:1:D
            prod1 = prod1*((99+x(i))/100)*cos(pi*x(i)^3);
        end
        fit = -prod1;
    case {38}, %Alpine
        sum1 = 0;
        for i=1:1:D
            sum1 = sum1 + abs(x(i)*sin(x(i)) + 0.1*x(i));
        end
        fit = sum1;
    case {39}, % Branin
        fit = (x(2)-(5.1/(4*pi^2))*x(1)^2+5*x(1)/pi-6)^2+10*(1-1/(8*pi))*cos(x(1))+10;
    case {40}, % Easom
        fit = -cos(x(1))*cos(x(2))*exp(-((x(1) - pi)^2 + (x(2) - pi)^2));
    case {41}, % Goldstein and Price
        fit = (1+(x(1) + x(2) + 1)^2*(19-14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2)^2))*(30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*x(1)^2+48*x(2)-36*x(1)*x(2)+27*x(2)^2));
    case {42}, % Shubert
        sum0 = 0;
        sum1 = 0;
        for j=1:1:5
            sum0 = sum0 + j*cos((j+1)*x(1) + j);
            sum1 = sum1 + j*cos((j+1)*x(2) + j);
        end
        fit = sum0*sum1;
    case {43}, % Hartmann
        a = [3 10 30; 0.1 10 35; 3 10 30; 0.1 10 35];
        c = [1 1.2 3 3.2];
        p = [0.3689 0.117 0.2673; 0.4699 0.4387 0.747; 0.1091 0.8732 0.5547; 0.0381 0.5743 0.8828];
        
        sum1 = 0;
        for i=1:1:4
            
            sum0 = 0;
            
            for j=1:1:3
                sum0 = sum0 + a(i,j)*(x(j) - p(i,j))^2;
            end
            
            sum1 = sum1 + c(i)*exp(-sum0);
        end
        fit = -sum1;
    case {44}, % Shekel function
% Matlab Code by A. Hedar (Nov. 23, 2005).
% The number of variables n = 4
% The parameter m should be adjusted m = 5,7,10.
% The default value of m = 10.
% 
m = 10;
a = ones(10,4);
a(1,:) = 4.0*a(1,:);
a(2,:) = 1.0*a(2,:);
a(3,:) = 8.0*a(3,:);
a(4,:) = 6.0*a(4,:);
for j = 1:2;
   a(5,2*j-1) = 3.0; a(5,2*j) = 7.0; 
   a(6,2*j-1) = 2.0; a(6,2*j) = 9.0; 
   a(7,j)     = 5.0; a(7,j+2) = 3.0;
   a(8,2*j-1) = 8.0; a(8,2*j) = 1.0;
   a(9,2*j-1) = 6.0; a(9,2*j) = 2.0;
   a(10,2*j-1)= 7.0; a(10,2*j)= 3.6;
end
c(1) = 0.1; c(2) = 0.2; c(3) = 0.2; c(4) = 0.4; c(5) = 0.4;
c(6) = 0.6; c(7) = 0.3; c(8) = 0.7; c(9) = 0.5; c(10)= 0.5;
s = 0;
for j = 1:m;
   p = 0;
   for i = 1:4
      p = p+(x(i)-a(j,i))^2;
   end
   s = s+1/(p+c(j));
end
fit = -s;
    case {45}, % Levy
n = D;
for i = 1:n; z(i) = 1+(x(i)-1)/4; end
s = sin(pi*z(1))^2;
for i = 1:n-1
    s = s+(z(i)-1)^2*(1+10*(sin(pi*z(i)+1))^2);
end 
fit = s+(z(n)-1)^2*(1+(sin(2*pi*z(n)))^2);    
    case {46}, % Michalewicz
        n = D; 
m = 10;
s = 0;
for i = 1:n;
    s = s+sin(x(i))*(sin(i*x(i)^2/pi))^(2*m);
end
fit = -s;

    case {47}, % Shifted Griewank
        
        sum1 = 0;
        sum2 = 1;
        
       f = -180;
       
       offset_5 = [-2.7626840e+002 -1.1911000e+001 -5.7878840e+002 -2.8764860e+002 -8.4385800e+001 -2.2867530e+002 -4.5815160e+002 -2.0221450e+002 -1.0586420e+002 -9.6489800e+001 -3.9574680e+002 -5.7294980e+002 -2.7036410e+002 -5.6685430e+002 -1.5242040e+002 -5.8838190e+002 -2.8288920e+002 -4.8888650e+002 -3.4698170e+002 -4.5304470e+002 -5.0658570e+002 -4.7599870e+002 -3.6204920e+002 -2.3323670e+002 -4.9198640e+002 -5.4408980e+002 -7.3445600e+001 -5.2690110e+002 -5.0225610e+002 -5.3723530e+002];
       
       for d=1:1:D
           xd = x(d) - offset_5(d);
           sum1 = sum1 + xd*xd;
           sum2 = sum2 * cos(xd/sqrt(d));
       end
       
       fit = f + 1.0 + sum1/4000.0 - sum2;
       
    case {48}, % Gear Train
    
        y = floor(x);
    
        fit = ((1/6.931) - ((y(1)*y(2))/(y(3)*y(4))))^2;
    
        for i = 1:1:4
            if y(i) > 60 || y(i) < 12
                fit = fit + 1e6;
            end
        end
    
    case {49}, % Pressure Vessel
    
        qd = 0.0625; % Granularity is 0.0625
        
        y1 = qd*floor(0.5 + x(1)/qd);
        
        y2 = qd*floor(0.5 + x(2)/qd);
    
        fit = 0.6224*y1*x(3)*x(4) + 1.7781*y2*x(3)^2 + 3.1661*y1^2*x(4) + 19.84*y1^2*x(3);
    
        if (-y1 + 0.0193*x(3)) > 0
       % disp('1x');
            fit = fit + 1e6;
        end
    
        if (-y2 + 0.00954*x(3)) > 0
        %  disp('2x');
            fit = fit + 1e6;
        end
    
        if (-pi*x(3)^2*x(4) - (4/3)*pi*x(3)^3 + 1296000) > 0
      %  disp('3x');
            fit = fit + 1e6;
        end
    
%         if (x(4) - 240) > 0
%        % disp('4x');
%             fit = fit + 1e6;
%         end
%     
%         if (1.1 - y1) > 0
%        % disp('4x');
%             fit = fit + 1e6;
%         end
%     
%         if (0.6 - y2) > 0
%        % disp('4x');
%             fit = fit + 1e6;
%         end
%         
    case {50}, % Tripod
        if x(1) <= 0
            sig_x1 = -1;
        else
            sig_x1 = 1;
        end
        
        if x(2) <= 0
            sig_x2 = -1;
        else
            sig_x2 = 1;
        end
        
        fit = 0.5*(1 - sig_x2)*(abs(x(1)) + abs(x(2) + 50));
        fit = fit + 0.5*(1 + sig_x2)*((0.5*(1 - sig_x1))*(1 + abs(x(1) + 50) + abs(x(2) - 50)) + (0.5*(1 + sig_x1))*(2 + abs(x(1) - 50) + abs(x(2) - 50)));
        
    case {51}, % Compression Spring
        
        y1 = floor(x(1));
        
        y2 = x(2);
        
        qd = 0.001; % Granularity is 0.001
        
        y3 = qd*floor(0.5 + x(3)/qd);
        
        Cf = 1 + 0.75*(y3/(y2 - y3)) + 0.615*(y3/y2);
        Fmax = 1000;
        S = 189000;
        K = 11.5*1e6*(y3^4/(8*y1*y2^3));
        lf = Fmax/K + 1.05*(y1 + 2)*y3;
        lmax = 14;
        Fp = 300;
        s_p = Fp/K;
        s_pm = 6;
        s_w = 1.25;
        
        fit = pi*pi*y2*y3*y3*(y1 + 2)*0.25; %(pi^2)*((y2*(y3^2)*(y1 + 1))/4);
        
        if ((8*Cf*Fmax*y2)/(pi*y3^3)) - S > 0
            fit = fit + 1e6;
        end
        
        if (lf - lmax) > 0
            fit = fit + 1e6;
        end
        
        if (s_p - s_pm) > 0
            fit = fit + 1e6;
        end
        
        if (s_p - Fp/K) > 0
            fit = fit + 1e6;
        end
        
        if (s_w - ((Fmax - Fp)/K)) > 0
            fit = fit + 1e6;
        end
            
       
    otherwise,
        error('No such function');
        fit = -1;
end

end

