% Dynamic Bayesian Network for phytoplankton - zooplankton
% dynamics in Archipelago sea. 

% Rasmus Boman 2020
% rasmus.a.boman@gmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%
% This model is a simple Naive Bayes that includes

% - phytoplankton in 7 variables (sorted by ~class)
% - zooplankton in 3 variables 
% - includes 1 hidden variables (1 generic)

% --> Hidden variable linked between time slices.

% Original variables in R:
% [1] "season"           "dis_org_nitr"     "dis_org_pho"      "salinity"         "temperature"     
% [6] "hvgen"            "Diatomophyceae"   "Dinophyceae"      "Litostomatea"     "Cyanophyceae"    
%[11] "Cryptophyceae"    "Chrysophyceae"    "Prymnesiophyceae" "Copepods"         "Cladocerans"     
%[16] "Synchaeta_sp"    

N = 16; % Number of nodes in the model

% Naming the variables for clarity

%%%%% NOTE TO ALLAN 01.07.2020
Season = 1; % Season should be a discrete variable in these models
%%%%% NOTE ENDS

DON = 2; DOP = 3; Sal = 4; Temp = 5; HVGen = 6; % Environmental & general HV

Diatom = 7; Dino = 8; Lito = 9; Cyano = 10; % Phytoplankton 1/2
Crypto = 11; Chryso = 12; Prymne = 13; % Phytoplankton 2/2 

Cope = 14; Clado = 15; Synch = 16; 

% DAG Structure

% Based on email discussion 15.7.2020 hidden variable linked only to
% plankton species.

ss = 16;
intra = zeros(N); % Create empty array for links

intra(Season, 7:16) = true; % Season to explain larger variability
intra(HVGen, 7:16) = true; % Generic HV -> all plankton

% No further links between variables in Naive Bayes

inter = zeros(N);
inter(HVGen, HVGen) = true;

% Read in the data
% Missing values encoded as NaN, converted to empty cell

data = readmatrix('Data/Seili_SLICED_log_scaled.csv'); 
data = num2cell(data);
[datlen, datn] = size(data);
for i = 1:datlen
    for j = 1:datn
        if isnan(data{i, j})
            data{i,j} = [];
        end
    end
end


% Which nodes will be observed? 
% Hidden variables will not be observed (variable 6)

onodes = [7:16]; 
dnodes = []; % Season should be a discrete node! (1)
ns = ones(1,N);

% Define equivalence classes for the model variables:
% Equivalence classes are needed in order to learn the conditional
% probability tables from the data so that all data related to a variable,
% i.e. data from all years, is used to learn the distribution; the eclass
% specifies which variables "are the same".

% In the first year, all vars have their own eclasses;
% in the consecutive years, each variable belongs to the same eclass 
% with itself from the other time slices. 
% This is because due to the temporal dependencies, some of the variables have a
% different number of incoming arcs, and therefore cannot be in the same
% eclass. 

eclass1 = 1:N; % first time slice
eclass2 = (N+1):(2*N);% consecutive time slices
eclass = [eclass1 eclass2];
 
% Make the model
bnet = mk_dbn(intra, inter, ns, 'observed', onodes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

%
% Loop over the EM learning 100 times, keep the best model (based on
% log-likelihood), to avoid getting a model that has got stuck to a poor local optimum
rng(13,'twister') % init the random number generator based on time stamp / changed to twister on 09.08.2019 (error) 
bestloglik = -inf; % initialize

for j = 1:100 % 
    j

    % Set the priors N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    % Junction tree learning engine for parameter learning
    
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500); 
    loglik = LLtrace(length(LLtrace));
    
    %when a better model is found, store it
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
            
    end
end

%save the bestbnet object
save('bestbnet_by_class')


t = datlen;

% mean and sd of each of the vars, each time slice
margs=SampleMarg(bestbnet,data(1:datlen,:)',t);

HVGenMu =[];% Generic HV standard deviation

HVGenSigma = []; %Generic HV mean
     
% write the means (Mu) and sds (sigma) of the interest variables down for easier access
for i = 1:t
    i
    %means
    HVGenMu(i) = margs{6,i}.mu;
    
    %sigma
    HVGenSigma(i) = margs{6,i}.Sigma;
end

    %save variables for plotting in R
    save('Results/Seili_Naive_Bayes_Mean.txt','HVGenMu','-ascii')
    
    save('Results/Seili_Naive_Bayes_Sigma.txt','HVGenSigma','-ascii')
