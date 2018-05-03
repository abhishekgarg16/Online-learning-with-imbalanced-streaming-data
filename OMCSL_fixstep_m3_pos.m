function [model, hat_y_t_pred, f_t_pred, l_t] = OMCSL_fixstep_m3_pos(y_t, x_t, model)
% Online Multiple Cost-Sensitive Learning: receive one data-label and update
% INPUT:
%   y_t:    a received ground-truth label, a scalar
%   x_t:    a received data instance, d * 1 vector
%   model:  a structure with fields:
%           loss_type:      1 -> hinge loss
%                           2 -> logistic loss
%                           3 -> squared loss
%           measure_type    1 -> F-measure
%                           2 -> AUROC
%                           3 -> AUPRC
%           K:              number of classifiers
%           prob:           probability, K * 1 array
%           ws:             (column) weight vectors of K classifiers, d * K matrix
%           bs:             bias terms of K classifiers, K * 1 array
%           measures:       performance measures of K classifiers, K * 1 array
%           gamma:          learning rate parameter
%           lambda:         strong convexity parameter (if strong convex)
%           t:              index of current iteration
%           aux:            auxiliary structure for incremental updating measeures with following fields
%                           1. F-measure
%                               1. as: K * 1 array
%                               2. cs: K * 1 array
%                           2. AUROC
%                               1. m:           scalar, number of ranges
%                               2. N_t_plus:    K * 1 array, # of positive instances within t iterations
%                               3. N_t_minor:   K * 1 array, # of negative instances within t iterations
%                               4. L_t_plus:    m * K array, hash table recording # of positive instances in each of m ranges for each of K classifiers 
%                               5. L_t_minor:   m * K array, hash table recording # of negative instances in each of m ranges for each of K classifiers
%                           3. AUPRC
%                               1. m:           scalar, number of ranges
%                               2. N_t_plus:    K * 1 array, # of positive instances within t iterations
%                               3. N_t_minor:   K * 1 array, # of negative instances within t iterations
%                               4. L_t_plus:    m * K array, hash table recording # of positive instances in each of m ranges for each of K classifiers 
%                               5. L_t_minor:   m * K array, hash table recording # of negative instances in each of m ranges for each of K classifiers
% OUTPUT: 
%    model

%% Initialization
x_t = x_t(:);

loss_type = model.loss_type; % string, loss function: 1 -> 'hinge';; 2 -> 'logistic';; 3 -> 'squared loss'
measure_type = model.measure_type; % string, measure types: 'F-measure', 'AUROC' and 'AUPRC'
K = model.K; % scalar, number of classifiers
prob = model.prob; % probability, K * 1 array
ws = model.ws; % weights of classifiers, d * K matrix (each w is a column vector)
bs = model.bs; % bias terms of classifiers, K * 1 array
accumulate_measures = model.accumulate_measures;
cost_pos = model.cost_pos;
cost_neg = model.cost_neg;

if isfield(model, 'measures')
    measures = model.measures; % measures, e.g. F-measure AUROC and AUPRC, a K * 1 vector
end

gamma = model.gamma; % learning rate parameter
t = model.t; % index of iteration
aux = model.aux; % auxiliary variables for incremental update of performance measures

% if isfield(model, 'lambda')
%     % strongly convex
%     eta_t = 1 / (model.lambda * t); % eta_t is the stepsize of OGD at t-th iteration
% else
%     % NOT strongly convex
%     eta_t = model.step_factor / sqrt(t); % eta_t is the stepsize of OGD at t-th iteration
% end
eta_t = model.step_factor;

%% Sample a classifier and predict
% [sid] = sample_classfier(prob);
try
if sum(isinf(prob)) == 0 && sum(isnan(prob)) == 0 && sum(prob < 0) == 0 && sum(prob > 0) > 0
    [sind] = randsample([1:K], 1, true, prob);
else
    [~,sind] = max(accumulate_measures(:,end));
end
catch
    disp(prob)
end

% hat_y_t = predict_one(ws(sid,:), bs(sid), x_t, loss_type);
f_t_pred = x_t' * ws(:,sind) + bs(sind);
if (f_t_pred > 0)
    hat_y_t_pred = 1;
else
    hat_y_t_pred = -1;
end

%% Update all K classifiers and their corresponding measures
for j = 1:K
    f_t = x_t' * ws(:,j) + bs(j);
    % 1. Update models, including \bw and b
    
    switch loss_type      
        case 1 % hinge loss
            l_t = max(0,1 - y_t * f_t);
            if (l_t > 0)            
                if y_t == 1
%                     cost = j / (K+1);
%                     cost = 0.5 + j/2/(K+1);
                    ws(:,j) = ws(:,j) + eta_t * cost_pos(j) * y_t * x_t;  % eta_t is the stepsize, and cost is the weight 
                    bs(j) = bs(j) + eta_t * cost_pos(j) * y_t; % eta_t is the stepsize, and cost is the weight                    
                else  % if y_t == -1
%                     cost = (K+1-j) / (K+1);
%                     cost = 0.5 - j/2/(K+1);
                    ws(:,j) = ws(:,j) + eta_t * cost_neg(j) * y_t * x_t;  % eta_t is the stepsize, and cost is the weight 
                    bs(j) = bs(j) + eta_t * cost_neg(j) * y_t;  % eta_t is the stepsize, and cost is the weight                 
                end
%                 ws(:,j) = ws(:,j) + eta_t * y_t * x_t;
%                 bs(j) = bs(j) + eta_t * y_t;
            end        
        case 2 % logistic loss
            l_t = log(1+exp(-y_t * f_t));
            if (l_t > 0)
                if y_t == 1
%                     cost = j / (K+1);
%                     cost = 0.5 + j/2/(K+1);
                    ws(:,j) = ws(:,j) + eta_t * cost_pos(j) * y_t * x_t * (1 / ( 1 + exp(y_t * f_t)));  % eta_t is the stepsize, and cost is the weight 
                    bs(j) = bs(j) + eta_t * cost_pos(j) * y_t * (1 / (1 + exp(y_t * f_t)));  % eta_t is the stepsize, and cost is the weight 
                else % if y_t == -1
%                     cost = (K+1-j) / (K+1);
%                     cost = 0.5 - j/2/(K+1);
                    ws(:,j) = ws(:,j) + eta_t * cost_neg(j) * y_t * x_t * (1 / ( 1 + exp(y_t * f_t)));  % eta_t is the stepsize, and cost is the weight 
                    bs(j) = bs(j) + eta_t * cost_neg(j) * y_t * (1 / (1 + exp(y_t * f_t)));  % eta_t is the stepsize, and cost is the weight                     
                end
%                 ws(:,j) = ws(:,j) + eta_t * cost * y_t * x_t * (1 / ( 1 + exp(y_t * f_t)));  % eta_t is the stepsize, and cost is the weight 
%                 bs(j) = bs(j) + eta_t * cost * y_t * (1 / (1 + exp(y_t * f_t)));  % eta_t is the stepsize, and cost is the weight 
            end        
        case 3 % square loss 
            l_t = 0.5*(y_t - f_t)^2;
            if (l_t > 0)
                ws(:,j) = ws(:,j) - eta_t * (f_t - y_t) * x_t;
                bs(j) = bs(j) - eta_t * (f_t - y_t);
            end
        otherwise
            error('Invalid loss type.');
    end
            
    
%     [ws(j,:)] = update_w_one(ws(j,:), x_t, y_t);
%     [bs(j)] = update_b_one(bs(j), x_t, y_t);
    
    % 2. Update measures for all classifiers (indenpendently)
%     [measures, as(j), cs(j)] = update_measure(measures, y_t, f_t, hat_y_t, measure_type, aux);
    switch measure_type
        case 1 % --- F-measure
            % 1. Update a_t^j
%             if y_t * f_t > 0
            if y_t == 1 && f_t > 0  % by yanyan: this may be better?
                aux.as(j) = aux.as(j) + 1;
            end
            
            % 2. Update c_t^j
            if y_t == 1 && f_t > 0
                aux.cs(j) = aux.cs(j) + 2;
            elseif y_t == 1 || f_t > 0  % this condition should be xor
                aux.cs(j) = aux.cs(j) + 1;
            elseif y_t == -1 && f_t <= 0
                aux.cs(j) = aux.cs(j);
            else
                % this condition should not happen if correct
                error('An unexpected case occures when updating c_t^j')
            end
            
            % 3. Update F-measure_t^j
            measures(j) = 2 * aux.as(j) / aux.cs(j);
            
                
            if measures(j) == Inf || isnan(measures(j))
                measures(j) = 0;
            end
            
        case 2 % --- AUROC
            if y_t == 1 % condition 1: a postive instance 
                sigmoid_f_t = 1/(1 + exp(-f_t));
                
                % find i which is the largest index such that i/m < sigmoid_f_t
                i = ceil(sigmoid_f_t * aux.m);
                
                % when sigmoid_f_t == 0, then we need to adjust i as follow
                if i == 0
                    i = i + 1;
                end
                
                % use i to retrieve L_t_minor[j] and L_t_minor[i+1]
%                 measures(j) = (aux.N_t_plus(j)/(aux.N_t_plus(j) + 1)) * measures(j) + (1/((aux.N_t_plus(j)+1)*aux.N_t_minor(j))) * (sum(aux.L_t_minor(1:i, j)) + aux.L_t_minor(i+1, j)/2);
                measures(j) = (aux.N_t_plus(j)/(aux.N_t_plus(j) + 1)) * measures(j) + (1/((aux.N_t_plus(j)+1)*aux.N_t_minor(j))) * (sum(aux.L_t_minor(1:i-1, j)) + aux.L_t_minor(i, j)/2);
                
                % Update L_t_plus[i+1] (since y_t == 1, we do not need to update L_t_minor)
                aux.L_t_plus(i, j) = aux.L_t_plus(i, j) + 1;
                
                % Update counter N_t_plus (since y_t == 1, we do not need to update N_t_minor)
                aux.N_t_plus(j) = aux.N_t_plus(j) + 1;
                
                if measures(j) == Inf || isnan(measures(j))
                    measures(j) = 0;
                end
            else
                % condition 2: a negative instance
                sigmoid_f_t = 1/(1 + exp(-f_t));
                
                % find i which is the smallest index such that i/m >= sigmoid_f_t
                i = ceil(sigmoid_f_t * aux.m);
                
                % when sigmoid_f_t == 0, then we need to adjust i as follow
                if i == 0
                    i = i + 1;
                end
                
                % use i to retrieve L_t_plus[j] and L_t_plus[i]
                measures(j) = (aux.N_t_minor(j)/(aux.N_t_minor(j)+1)) * measures(j) + (1/(aux.N_t_plus(j)*(aux.N_t_minor(j)+1))) * (sum(aux.L_t_plus(i+1:end, j)) + aux.L_t_plus(i, j)/2);
                
                % update L_t_minor[i] (since y_t == -1, we do not need to update L_t_plus)
                aux.L_t_minor(i, j) = aux.L_t_minor(i, j) + 1;
                
                % update counter N_t_minor (since Y_t == -1, we do not need to update N_t_plus)
                aux.N_t_minor(j) = aux.N_t_minor(j) + 1;
                
                if measures(j) == Inf || isnan(measures(j))
                    measures(j) = 0;
                end
            end
            
        case 3 % --- AUPRC
            
            % Update N_t_plus/N_t_minor and L_t_plus/L_t_minor
            if y_t == 1
                aux.N_t_plus(j) = aux.N_t_plus(j) + 1;
                
                sigmoid_f_t = 1/(1 + exp(-f_t));
                
                % find i which is the smallest index such that i/m >= sigmoid_f_t
                i = ceil(sigmoid_f_t * aux.m);
                
                % when sigmoid_f_t == 0, then we need to adjust i as follow
                if i == 0
                    i = i + 1;
                end
                aux.L_t_plus(i, j) = aux.L_t_plus(i, j) + 1;
            elseif y_t == -1
                aux.N_t_minor(j) = aux.N_t_minor(j) + 1;
                sigmoid_f_t = 1/(1 + exp(-f_t));
                
                % find i which is the smallest index such that i/m >= sigmoid_f_t
                i = ceil(sigmoid_f_t * aux.m);
                
                % when sigmoid_f_t == 0, then we need to adjust i as follow
                if i == 0
                    i = i + 1;
                end
                
                % update L_t_minor[i] (since y_t == -1, we do not need to update L_t_plus)
                aux.L_t_minor(i, j) = aux.L_t_minor(i, j) + 1;
            else
                error('Unknown class label when updating online measure!')
            end
            
            % Compute RR(i) = R_t(i-1) - R_t(i) = L_t_plus(i)/N_t_plus for i=1:m
            RR = aux.L_t_plus(:,j) ./ aux.N_t_plus(j);
            
            % and compute PP(i) = P_t(i-1) + P_t(i) for i=1:m
            P_t = zeros(aux.m+1,1);
            P_t_ind1 = intersect(find(RR ~= 0), find(~isnan(RR)));
%             P_t_ind2 = union(P_t_ind1, P_t_ind1 + 1);
            P_t_ind2 = union(P_t_ind1, P_t_ind1);
            P_t_ind = union(P_t_ind2, P_t_ind1 - 1);
            for i_m=P_t_ind'
%             for i_m=0:aux.m
                P_t(i_m+1) = (sum(aux.L_t_plus(i_m+1:aux.m,j)) / (sum(aux.L_t_plus(i_m+1:aux.m,j)) + sum(aux.L_t_minor(i_m+1:aux.m,j))));
%                 P_t(i_m+1) = (sum(aux.L_t_plus(i_m:aux.m,j)) / (sum(aux.L_t_plus(i_m:aux.m,j)) + sum(aux.L_t_minor(i_m:aux.m,j))));
            end
            P_t(isnan(P_t)) = 0;
            PP = P_t(1:end-1) + P_t(2:end);
            
            measures(j) = sum(RR .* PP) / 2;
            
            if measures(j) == Inf || isnan(measures(j))
                measures(j) = 0;
            end
            
    end
end


%% Update the probability
% [prob] = update_prob(measures, gamma);
sum_measures = sum(exp(gamma .* measures));
for j = 1:K
    prob(j) = exp(gamma * measures(j)) / sum_measures;
end


%% Glue everything back to model
model.prob = prob; % probability, K * 1 array
model.ws = ws; % weights of classifiers, K * d matrix (each w is a column vector)
model.bs = bs; % bias terms of classifiers, K * 1 array
model.measures = measures; % measures, e.g. F-measure AUROC and AUPRC
model.accumulate_measures = [model.accumulate_measures,measures;];
model.sind = [model.sind; sind];
model.t = t + 1;

model.aux = aux;
model.f_t = f_t;  % predict value

end % the end of the function

