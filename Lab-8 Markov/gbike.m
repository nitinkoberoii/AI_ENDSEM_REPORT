% Gbike Bicycle Rental Problem: Policy Iteration

% Main script for Policy Iteration Algorithm
clear;
clc;

% Parameters
Lamda = [3, 4]; % Expected rental requests (Poisson distribution)
lamda = [3, 2]; % Expected returns (Poisson distribution)
r = 10;  % Revenue per bike rented
t = 2;   % Cost per bike moved
gam = 0.9; % Discount factor
max_bikes = 20; % Max bikes at each location
max_move = 5;  % Max bikes that can be moved overnight
theta = 0.01;  % Convergence threshold for policy evaluation
max_iter = 1000; % Maximum iterations for policy evaluation

% State space
m = max_bikes + 1; % Number of states for location 1
n = max_bikes + 1; % Number of states for location 2
policy = zeros(m, n); % Policy (action to take)
V = zeros(m, n); % Value function

% Initialize policy randomly
policy(:) = randi([-max_move, max_move], m, n);

% Run policy iteration
[policy, policy_stable] = policy_iteration_gbike(V, policy, Lamda, lamda, r, t, gam, m, n, theta, max_iter, max_bikes);

% Display results
disp('Optimal Policy:');
disp(policy);
disp('Optimal Value Function:');
disp(V);

% Visualize Optimal Policy
figure;
imagesc(policy);  % Create heatmap for policy
colorbar;         % Add colorbar
title('Optimal Policy');
xlabel('Location 2 Bikes');
ylabel('Location 1 Bikes');
axis tight;

% Visualize Optimal Value Function
figure;
surf(V);  % Create 3D surface plot for value function
colorbar; % Add colorbar
title('Optimal Value Function');
xlabel('Location 2 Bikes');
ylabel('Location 1 Bikes');
zlabel('Value');

%% Policy Iteration Function
function [policy, policy_stable] = policy_iteration_gbike(V, policy, Lamda, lamda, r, t, gam, m, n, theta, max_iter, max_bikes)
    policy_stable = true; % Initialize policy stability flag
    count = 0; % Iteration counter
    
    while true
        count = count + 1;
        disp(['Iteration ', num2str(count)]);
        
        % Policy Evaluation
        V = policy_evaluation_gbike(V, policy, Lamda, lamda, r, t, gam, m, n, theta, max_iter, max_bikes);
        
        % Policy Improvement
        [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam, m, n, max_bikes);
        
        % Check if policy has stabilized
        if policy_stable
            break;
        end
        
        if count > max_iter
            disp('Maximum iterations reached!');
            break;
        end
    end
end

%% Policy Evaluation Function
function V = policy_evaluation_gbike(V, policy, Lamda, lamda, r, t, gam, m, n, theta, max_iter, max_bikes)
    max_iter_eval = 1000; % Max iterations for policy evaluation
    delta = Inf;
    iter_count = 0;
    
    % Define Poisson distribution probabilities for requests and returns
    P1 = poisson_dist(0:max_bikes, Lamda(1));  % Probability for rental requests at location 1
    P2 = poisson_dist(0:max_bikes, Lamda(2));  % Probability for rental requests at location 2
    P3 = poisson_dist(0:max_bikes, lamda(1));  % Probability for returns at location 1
    P4 = poisson_dist(0:max_bikes, lamda(2));  % Probability for returns at location 2
    
    while delta > theta && iter_count < max_iter_eval
        iter_count = iter_count + 1;
        v = V; % Save old value function
        
        for i = 1:m
            for j = 1:n
                s1 = i - 1; s2 = j - 1; % State (0-20, 0-20)
                a = policy(i, j); % Action (number of bikes moved)
                Vs_ = 0;
                
                % Calculate reward and transition
                R = -abs(a) * t; % Expected reward from state (s1, s2)
                s1_ = s1 - a; s2_ = s2 + a;
                
                for n1 = 0:max_bikes
                    for n2 = 0:max_bikes
                        s1__ = s1_ - min(n1, s1_);
                        s2__ = s2_ - min(n2, s2_);
                        
                        for n3 = 0:max_bikes
                            for n4 = 0:max_bikes
                                s1___ = s1__ + min(n3, max_bikes - s1__);
                                s2___ = s2__ + min(n4, max_bikes - s2__);
                                
                                % Value of future states
                                Vs_ = Vs_ + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * V(s1___ + 1, s2___ + 1));
                                
                                % Calculate reward
                                R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * (min(n1, s1_) + min(n2, s2_))) * r;
                            end
                        end
                    end
                end
                
                V(i, j) = R + (gam * Vs_);
            end
        end
        
        delta = max(max(abs(v - V))); % Convergence check
        disp(['Policy Evaluation Iteration: ', num2str(iter_count), ', Delta: ', num2str(delta)]);
        
        if iter_count > max_iter_eval
            disp('Maximum policy evaluation iterations reached!');
            break;
        end
    end
end

%% Policy Improvement Function
function [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam, m, n, max_bikes)
    policy_stable = true;
    
    % Define Poisson distribution probabilities for requests and returns
    P1 = poisson_dist(0:max_bikes, Lamda(1));  % Probability for rental requests at location 1
    P2 = poisson_dist(0:max_bikes, Lamda(2));  % Probability for rental requests at location 2
    P3 = poisson_dist(0:max_bikes, lamda(1));  % Probability for returns at location 1
    P4 = poisson_dist(0:max_bikes, lamda(2));  % Probability for returns at location 2
    
    for i = 1:m
        for j = 1:n
            s1 = i - 1; s2 = j - 1; % State (0-20, 0-20)
            amin = -min(min(s2, m - 1 - s1), 5);
            amax = min(min(s1, n - 1 - s2), 5);
            v_ = -Inf;
            
            for a = amin:amax
                R = -abs(a) * t; % Expected reward starting from state (s1, s2)
                Vs_ = 0;
                s1_ = s1 - a; s2_ = s2 + a;
                
                for n1 = 0:max_bikes
                    for n2 = 0:max_bikes
                        s1__ = s1_ - min(n1, s1_);
                        s2__ = s2_ - min(n2, s2_);
                        
                        for n3 = 0:max_bikes
                            for n4 = 0:max_bikes
                                s1___ = s1__ + min(n3, max_bikes - s1__);
                                s2___ = s2__ + min(n4, max_bikes - s2__);
                                Vs_ = Vs_ + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * V(s1___ + 1, s2___ + 1));
                                
                                % Add reward
                                R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * (min(n1, s1_) + min(n2, s2_))) * r;
                            end
                        end
                    end
                end
                
                % Update policy with the action that maximizes value
                if (R + gam * Vs_ > v_)
                    v_ = R + gam * Vs_;
                    policy(i, j) = a;
                end
            end
        end
    end
end

%% Poisson Distribution Function
function P = poisson_dist(x, lambda)
    P = (lambda .^ x) .* exp(-lambda) ./ factorial(x);
end
