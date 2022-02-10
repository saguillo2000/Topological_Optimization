"""
TODO

X hashing and memos

X correct typos

X other vector quantizations (harmonic to minimum lvq in particular)

X refactor metrics and clusterings to eschew point data

X refactor clusterings again to process tensors (as opposed to ndarrays)

refactor clusterings again again to process tensors sensibly (as opposed to casting to and from ndarray)
    this is especially urgent if array->tensor or tensor->array constructors aren't move constructors... so are they?
    update: numpy->tensor is fast, but tensor->numpy is slow.
        this does actually makes the issue pretty urgent, since the former is only applied to downsampled tensors
    update: is this even possible?
        X random
        X maxmin
        X thresholded maxmim
        frechet
        vq
        harmonic
    do the same to metrics, aswell
        X Metric
        X CompleteManifold
        CorrelationSimilarity
        EuclideanCorrelationMetric
        X EuclideanMetric
        ProjectiveCorrelationSimilarity
        ProjectiveEuclideanCorrelationMetric
        X ProjectiveSphereCorrelationMetric
        X ProjectiveSphereMetric
        X SphereCorrelationMetric
        X SphereMetric
    also vectorize reductions, you git
        X Metric
        CorrelationSimilarity
        EuclideanCorrelationMetric
        X EuclideanMetric
        ProjectiveCorrelationSimilarity
        ProjectiveEuclideanCorrelationMetric
        X ProjectiveSphereCorrelationMetric
        X ProjectiveSphereMetric
        X SphereCorrelationMetric
        X SphereMetric

N density estimation

X thresholded maxmin

X exponential, logarithmic map in CompleteManifold

X differentiable-ish K-means with frechet means

X pass seed kwarg to manifold samplers

    X avoid passing the same seed each time

N model to features standarization

X layer-to-layer correlation histograms

X keep filling after discards @ thresholded maxmin

X ensure memoing works w/ everything necessary (pickling tensors, ndarrays, persistence diagrams - hashing models, tensors, ndarrays)

X fix the vectorization wank @ manifold-dependent clustering methods

X either segregate matrix distance generation or edit the distance wrapper to optionally take 1 argument and build a triangular distance matrix

N pre-downsampling in the downsampling strategies that iterate over every point... in general, capping the execution time a bit

X vectorized reduction

X gstrat but with metrics

order (MI) fitting
    ambient space perturbation.reduction, as opposed to tangent gradient
    order-based error
    smooth kendall/MI based error

X isotropy check (add vs mul)

X fix vectorized reduction

X ensure manifold sampling accounts for irreducible elements

? test stuff
    X refactor for functionality
    X test w first two spaces, ensuring memos work
    X for every combination,
        metric = SphereCorrelationMetric, ProjectiveSphereCorrelationMetric
        clustering = random, maxmin, thresholdedmaxmin, frechet-k-means if it's not too slow
    N ruben-sponsored linear fitting
    while that executes, introduce the repeated-tda-then-averaging (on PI, PL or PD)
        seed as kwarg for sampling - or prebaked repeat sampling



X push stuff

? redo sampling_visualization.ipynb
    X clean up
    ? discuss PH-subsampling
    ? use it for some cool sampling tests

explore fixed homology concept

X metric summaries

X test metric summaries

X metric summaries from distance matrix

? cayley menger, but it pre-maxmins so that ambient dim = qt points

X topological summaries

X landscape norm (integral limits)

X pull diagrams from server

X write precomputation (of all summaries)

X push to server, run

X fix landscapes (something about strict inequalities)

X landscapes
    X write the read-write dict stuff
    X push to server
    X execute
    X do analysise

X MI modularization (is this possible?)

X compute correlation matrix (mb segregate by tasks, too)

X correlation fitting, relevancy partitioning

? look into the maxmin results

X faster PIs
    X fix PI edge cases
    X stop recomputing weights

R faster PLs

X PL product
    X fix
    now do it again but better (twin interval stepping)

X PD dimensions
    X weird stairs
    X sort by generalization, then watch stairs
    X plot some tentative summaries (restricted by dim)
    N test batch (jfc)

investigate simpler form
    "classical" predictors MI/corr box
        XX flatness
        XX gradient noise
        XX path norm
        XX various norms
        XX various spectral methods

    variance??
    ordered dists

X less dimension, more points
    X corr box again

X investigate ripser point limit

N demogen integration
N ripser++

X revisit dimension estimators

something something cayley menger rank (greedy?)

revise classical estimators
    spectral norms but fft iterates over layers so tf doesn't stb
    X frobenius conv vs linear operator
    gradient noise but it's element-wise

X follow up on dimension wank (particularly homology dimensions, as in "Fractal Dimension Estimation")
    X see if tda summaries correlate with ph-0 dim (i.e. if the amt of features is stable across models)

investigate dim/tda connection with generalization
    see if dimension (or, well, tda) correlates with any "classical" predictor

final MI box, si no tarda mil anys

temp
    X push
    X read from serv unpickling method using to avoid repeating estimator computations
    X introduce it, commented, in estimatorsprecomputation
    X introduce aswell if model_num==159: return (not commented)
    X scp over EstimatorsPrecomputation.py and NormBased.py
    X compute
    take exclusions into account for the statistical analysis

temp 2 (JUST task 1)
    N recompute local (model_20) (algo passa con les mappedmatrices)
    pull remote estimators somewhere
    merge them w/ local

(...)

em pregunto si hi ha una manera d'estimar dimensions a partir de la distribució de les cel·les de voronoi del maxmin
atés la forma de la cua inferior, més o menys, hauria de dependre de la mesura induida del marge/la mesura de la distribució
vaja, probablement depengui massa fortament de la geometria global...
(que, en certa manera, ho fa akin a les discretitzacions de minkowski. Potser sí que val la pena)

en retrospectiva, això de les manifolles té més a veure amb gauss-bonnet del que creia. De qui es la curvatura és indiferent-
la cosa es que una 1-manifolla (sigui o no geodèsica d'una hipotètica varietat superjacent) sempre admet una isometria amb
un espai pla (els reals). Que les 2-manifolles, curvatura extrínseca o no, no tinguin aquesta propietat (la desitjada
per fer sumes afins) es un dels primers resultats importants de gd.

la persistence surface es una convolució de la distribució per weight*D, ón D és una funció generalitzada obtinguda
d'una suma de deltes de dirac centrades al diagrama de persistència subjacent

els 4 summaries de PDS, restringits a dimensió, donen correlacions extranyes entre dimensions. I.e. = dimensió 0,2,3 estàn
bastant correlacionades (encara que 0 negativament amb les altres dues), pero 1 no té massa a veure. ...?
si es segrega en task1 i task2, task2 sembla tenir una correlació, però molt fluixa, mentre que task1...
sembla seguir una relació no lineal?

quines formes poden pendre els grups d'homologia que no siguin representades per els seus rangs?

qué es representat per els topological summaries, ABANS de la topologia?

qualificatius dels epsilons del maxmin

"""