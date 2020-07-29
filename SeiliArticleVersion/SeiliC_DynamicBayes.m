% Dynamic Bayesian Network for phytoplankton - zooplankton
% dynamics in Archipelago sea. 

% Rasmus Boman 2020
% rasmus.a.boman@gmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%
% This model is dynamic Bayes that includes

% - phytoplankton in 7 variables (sorted by ~class)
% - zooplankton in 3 variables 
% - 1 hidden variables (1 generic)

% --> Dynamic version, links between plankton groups and timeslices.

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

% "intras" are for one time slice

ss = 16;
intra = zeros(N); % Create empty array for links

% Environmental variables 

intra(Season, [2:5 7:16]) = true; % Season linked to all variables
intra(DON, 7:13) = true; %  Dissolved organic nitrogen -> phytoplankton
intra(DOP, 7:13) = true; % Dissolved organic phosphorus -> phytoplankton
intra(Sal, 7:16) = true; % salinity -> all plankton
intra(Temp, 7:16) = true; % temperature -> all plankton 

intra(HVGen, 7:16) = true; % Generic HV -> all plankton

% Phytoplankton layer
intra(7:9, 14) = true; % Diatom, Dino, Lito -> Copepods
intra([7:8 11:12], 15) = true; % Phyto (-Lito, -Cyano) -> Cladocerans
intra([7:8 11:12], 16) = true; % Phyto (-Lito, -Cyano) -> Synchaeta

%%% inter-dependencies %%%

% "inter" refers to the dependencies between time slices

inter = zeros(N); % table to build in the dependecies
inter(HVGen, HVGen) = true; % Hidden variable linked to itself

% temperature predicting next slice as well:
 inter(Temp, 7:16) = true;
% 
% Phytoplankton connections, previous stocks' effect on the next:
inter(Diatom, Diatom) = true; 
inter(Dino, Dino) = true;
inter(Lito, Lito) = true;
inter(Cyano, Cyano) = true;
inter(Crypto, Crypto) = true;
inter(Chryso, Chryso) = true;
inter(Prymne, Prymne) = true;

% Zooplankton's effect on itself
% Cope = 14; Clado = 15; Synch = 16;
inter(Cope, Cope) = true;
inter(Clado, Clado) = true;
inter(Synch, Synch) = true;

% Phytoplankton layer
inter(7:9, 14) = true; % Diatom, Dino, Lito -> Copepods
inter([7:8 11:12], 15) = true; % Phyto (-Lito, -Cyano) -> Cladocerans
inter([7:8 11:12], 16) = true; % Phyto (-Lito, -Cyano) -> Synchaeta

% Read in the data
% Missing values encoded as NaN, converted to empty cell
% The file needs to have the variables in the numbered order in columns!!
% Also HVs

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
% Hidden variables will not be observed (variables 5 and 13)

onodes = [2:5, 7:16]; 
dnodes = []; % Month is a discrete node
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


t = datlen; % ~30 years

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
    save('Results/Seili_Dynamic_Bayes_Mean.txt.txt','HVGenMu','-ascii')
    
    save('Results/Seili_Dynamic_Bayes_Sigma.txt.txt','HVGenSigma','-ascii')
