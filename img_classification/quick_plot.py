import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

data = {
    "Learned λ": [
        {"train_lr": 9.999999999999667e-07, "train_loss": 6.9127689444181675, "test_loss": 6.895549573971115, "test_acc1": 0.17600000701904298, "test_acc5": 0.7140000178527832, "epoch": 0},
        {"train_lr": 9.999999999999667e-07, "train_loss": 6.905367200158303, "test_loss": 6.882812117802278, "test_acc1": 0.21400000762939453, "test_acc5": 0.9360000305175781, "epoch": 1},
        {"train_lr": 5.0800000000000334e-05, "train_loss": 6.8654219074043965, "test_loss": 6.7134499185867895, "test_acc1": 0.5680000158691406, "test_acc5": 2.5400000871276855, "epoch": 2},
        {"train_lr": 0.00010060000000000406, "train_loss": 6.8099416760161615, "test_loss": 6.5384269961873995, "test_acc1": 1.2140000326538085, "test_acc5": 4.260000140533447, "epoch": 3},
        {"train_lr": 0.0001503999999999962, "train_loss": 6.729460735800082, "test_loss": 6.339160143874074, "test_acc1": 1.8000000686645508, "test_acc5": 6.358000184020996, "epoch": 4},
        {"train_lr": 0.00020019999999999728, "train_loss": 6.679011984521582, "test_loss": 6.201620218408017, "test_acc1": 2.146000065307617, "test_acc5": 7.46400020980835, "epoch": 5},
        {"train_lr": 0.0002339230484541401, "train_loss": 6.62472868332308, "test_loss": 6.052863597869873, "test_acc1": 2.752000082244873, "test_acc5": 9.482000295562743, "epoch": 6},
        {"train_lr": 0.00022708203932498923, "train_loss": 6.575409305089562, "test_loss": 5.955343002581414, "test_acc1": 3.6300001139831544, "test_acc5": 11.47200034362793, "epoch": 7},
        {"train_lr": 0.00021917737905729635, "train_loss": 6.526491606314936, "test_loss": 5.812581790312556, "test_acc1": 4.41200013092041, "test_acc5": 13.686000405883789, "epoch": 8},
        {"train_lr": 0.00021029567276305505, "train_loss": 6.487548200000194, "test_loss": 5.722878299596656, "test_acc1": 4.740000154266357, "test_acc5": 14.500000482177734, "epoch": 9},
        {"train_lr": 0.00020053423027510315, "train_loss": 6.449264575737309, "test_loss": 5.625275841196075, "test_acc1": 5.714000166320801, "test_acc5": 16.47800044708252, "epoch": 10},
        {"train_lr": 0.00019000000000000568, "train_loss": 6.405782027222571, "test_loss": 5.503197597183344, "test_acc1": 6.482000205078125, "test_acc5": 18.446000497436522, "epoch": 11},
        {"train_lr": 0.00017880839716910164, "train_loss": 6.376196088209206, "test_loss": 5.463807488215789, "test_acc1": 7.204000201263428, "test_acc5": 19.862000524902342, "epoch": 12},
        {"train_lr": 0.00016708203932499274, "train_loss": 6.333726508748159, "test_loss": 5.355666764819895, "test_acc1": 7.6900002394104, "test_acc5": 21.048000615234375, "epoch": 13},
        {"train_lr": 0.00015494940289812457, "train_loss": 6.297667558103998, "test_loss": 5.259360251535896, "test_acc1": 8.846000268249512, "test_acc5": 23.272000603027344, "epoch": 14},
        {"train_lr": 0.00014254341559211594, "train_loss": 6.276522905016361, "test_loss": 5.189223460568726, "test_acc1": 9.288000253295898, "test_acc5": 24.342000723876954, "epoch": 15},
        {"train_lr": 0.0001299999999999971, "train_loss": 6.246109543794977, "test_loss": 5.096829414367676, "test_acc1": 10.570000298156739, "test_acc5": 26.44400078491211, "epoch": 16},
        {"train_lr": 0.00011745658440787735, "train_loss": 6.211739843408613, "test_loss": 5.039882057495699, "test_acc1": 10.98400033203125, "test_acc5": 27.302000775146485, "epoch": 17},
        {"train_lr": 0.00010505059710186537, "train_loss": 6.1805536909509105, "test_loss": 4.98223698412189, "test_acc1": 11.634000299072266, "test_acc5": 28.604000793457033, "epoch": 18},
        {"train_lr": 9.291796067500908e-05, "train_loss": 6.148507775434282, "test_loss": 4.895383223322511, "test_acc1": 12.362000349121093, "test_acc5": 29.93000080810547, "epoch": 19},
        {"train_lr": 8.11916028309045e-05, "train_loss": 6.1329127694936, "test_loss": 4.873168852493053, "test_acc1": 12.366000365600586, "test_acc5": 30.200000808105468, "epoch": 20},
        {"train_lr": 6.99999999999994e-05, "train_loss": 6.105010492382509, "test_loss": 4.801553103759999, "test_acc1": 13.452000423583984, "test_acc5": 31.55600091064453, "epoch": 21},
        {"train_lr": 5.946576972490346e-05, "train_loss": 6.097099189328145, "test_loss": 4.746247934021112, "test_acc1": 13.92000038696289, "test_acc5": 32.48000084716797, "epoch": 22},
        {"train_lr": 4.9704327236939095e-05, "train_loss": 6.074022663611134, "test_loss": 4.731557316452492, "test_acc1": 14.290000380859375, "test_acc5": 33.30800093261719, "epoch": 23},
        {"train_lr": 4.0822620942711125e-05, "train_loss": 6.069144828205901, "test_loss": 4.6929793812846405, "test_acc1": 14.590000447998047, "test_acc5": 33.902000842285155, "epoch": 24},
        {"train_lr": 3.2917960675005526e-05, "train_loss": 6.0550569896512245, "test_loss": 4.6645494326380375, "test_acc1": 15.022000455932616, "test_acc5": 34.528001049804686, "epoch": 25},
        {"train_lr": 2.6076951545868032e-05, "train_loss": 6.041606232669035, "test_loss": 4.632097715639886, "test_acc1": 15.32200044128418, "test_acc5": 34.95400106201172, "epoch": 26},
        {"train_lr": 2.037454508288773e-05, "train_loss": 6.015849933692701, "test_loss": 4.6063030858076255, "test_acc1": 15.560000482177735, "test_acc5": 35.39200096923828, "epoch": 27},
        {"train_lr": 1.5873218044581917e-05, "train_loss": 6.013901184302974, "test_loss": 4.590430964040392, "test_acc1": 15.748000362548828, "test_acc5": 35.59000099609375, "epoch": 28},
        {"train_lr": 1.2622287911943719e-05, "train_loss": 5.994505577488229, "test_loss": 4.581982931107965, "test_acc1": 15.86800049621582, "test_acc5": 35.880000871582034, "epoch": 29},
    ],
    "Fixed λ=0.5": [
        {"epoch": 0, "train_loss": 6.912770659418854, "test_loss": 6.895597694484332, "test_acc1": 0.17200000762939452, "test_acc5": 0.7},
        {"epoch": 1, "train_loss": 6.90538254693372, "test_loss": 6.882855382584434, "test_acc1": 0.216, "test_acc5": 0.936},
        {"epoch": 2, "train_loss": 6.865964435174611, "test_loss": 6.717535674116994, "test_acc1": 0.56, "test_acc5": 2.488},
        {"epoch": 3, "train_loss": 6.813684532912434, "test_loss": 6.549346519790533, "test_acc1": 1.15, "test_acc5": 4.176},
        {"epoch": 4, "train_loss": 6.736773646592727, "test_loss": 6.360623836517334, "test_acc1": 1.728, "test_acc5": 6.19},
        {"epoch": 5, "train_loss": 6.682619237337645, "test_loss": 6.21754881807866, "test_acc1": 2.012, "test_acc5": 7.126},
        {"epoch": 6, "train_loss": 6.625810117980628, "test_loss": 6.0701343958614435, "test_acc1": 2.766, "test_acc5": 9.284},
        {"epoch": 7, "train_loss": 6.580353429658421, "test_loss": 5.974293504962485, "test_acc1": 3.504, "test_acc5": 11.102},
        {"epoch": 8, "train_loss": 6.538150726251881, "test_loss": 5.864061941627328, "test_acc1": 4.008, "test_acc5": 12.712},
        {"epoch": 9, "train_loss": 6.505797571673997, "test_loss": 5.776009745270241, "test_acc1": 4.268, "test_acc5": 13.462},
        {"epoch": 10, "train_loss": 6.4732573532190525, "test_loss": 5.695932690423864, "test_acc1": 5.262, "test_acc5": 15.146},
        {"epoch": 11, "train_loss": 6.4376568280996755, "test_loss": 5.612513094457961, "test_acc1": 5.422, "test_acc5": 16.268},
        {"epoch": 12, "train_loss": 6.412641990129792, "test_loss": 5.554817145107356, "test_acc1": 6.55, "test_acc5": 18.258},
        {"epoch": 13, "train_loss": 6.377133106464364, "test_loss": 5.477121720787223, "test_acc1": 6.734, "test_acc5": 18.906},
        {"epoch": 14, "train_loss": 6.348928706454107, "test_loss": 5.400245513624817, "test_acc1": 7.516, "test_acc5": 20.614},
        {"epoch": 15, "train_loss": 6.333471491787752, "test_loss": 5.34376816713173, "test_acc1": 7.984, "test_acc5": 21.758},
        {"epoch": 16, "train_loss": 6.30931374355807, "test_loss": 5.28376233122731, "test_acc1": 8.944, "test_acc5": 23.284},
        {"epoch": 17, "train_loss": 6.281956219783139, "test_loss": 5.23735299365211, "test_acc1": 9.126, "test_acc5": 23.668},
        {"epoch": 18, "train_loss": 6.257798034554564, "test_loss": 5.196662189396283, "test_acc1": 9.654, "test_acc5": 24.49},
        {"epoch": 19, "train_loss": 6.233206693726524, "test_loss": 5.129904535890535, "test_acc1": 10.23, "test_acc5": 25.51},
        {"epoch": 20, "train_loss": 6.2209165923599095, "test_loss": 5.1202391267732805, "test_acc1": 10.212, "test_acc5": 25.59},
        {"epoch": 21, "train_loss": 6.198551985008908, "test_loss": 5.048251101078878, "test_acc1": 11.092, "test_acc5": 27.07},
        {"epoch": 22, "train_loss": 6.192749764233721, "test_loss": 5.0058246623468765, "test_acc1": 11.444, "test_acc5": 27.784},
        {"epoch": 23, "train_loss": 6.174137742625695, "test_loss": 4.984643512099754, "test_acc1": 11.742, "test_acc5": 28.508},
        {"epoch": 24, "train_loss": 6.170355833474946, "test_loss": 4.96264814056513, "test_acc1": 12.052, "test_acc5": 28.926},
        {"epoch": 25, "train_loss": 6.1587934789750465, "test_loss": 4.948714178027087, "test_acc1": 12.248, "test_acc5": 29.314},
        {"epoch": 26, "train_loss": 6.147575160162929, "test_loss": 4.910027158169346, "test_acc1": 12.546, "test_acc5": 29.864},
        {"epoch": 27, "train_loss": 6.126665897212964, "test_loss": 4.892153173912573, "test_acc1": 12.676, "test_acc5": 30.022},
        {"epoch": 28, "train_loss": 6.126026976481271, "test_loss": 4.878462893362264, "test_acc1": 12.856, "test_acc5": 30.344},
        {"epoch": 29, "train_loss": 6.109211554104459, "test_loss": 4.871473603576194, "test_acc1": 12.956, "test_acc5": 30.506},
    ],
    "Fixed λ=0.8": [
        {"epoch": 0, "train_loss": 6.912779760140752, "test_loss": 6.8954906354423695, "test_acc1": 0.178, "test_acc5": 0.694},
        {"epoch": 1, "train_loss": 6.905172896959548, "test_loss": 6.882591309438225, "test_acc1": 0.218, "test_acc5": 0.926},
        {"epoch": 2, "train_loss": 6.864697241282109, "test_loss": 6.715329075587615, "test_acc1": 0.586, "test_acc5": 2.528},
        {"epoch": 3, "train_loss": 6.805998573909351, "test_loss": 6.518172085740184, "test_acc1": 1.24, "test_acc5": 4.652},
        {"epoch": 4, "train_loss": 6.725480608302224, "test_loss": 6.3276149334798335, "test_acc1": 1.8, "test_acc5": 6.54},
        {"epoch": 5, "train_loss": 6.669579749349324, "test_loss": 6.166311893754333, "test_acc1": 2.32, "test_acc5": 8.098},
        {"epoch": 6, "train_loss": 6.609308574701932, "test_loss": 6.010458989907767, "test_acc1": 3.022, "test_acc5": 9.96},
        {"epoch": 7, "train_loss": 6.561400677472246, "test_loss": 5.929701641315722, "test_acc1": 3.748, "test_acc5": 11.758},
        {"epoch": 8, "train_loss": 6.514275156370131, "test_loss": 5.8116597328477235, "test_acc1": 4.35, "test_acc5": 13.562},
        {"epoch": 9, "train_loss": 6.4767575992308535, "test_loss": 5.711505180096808, "test_acc1": 4.776, "test_acc5": 14.788},
        {"epoch": 10, "train_loss": 6.44030542363881, "test_loss": 5.607456429314067, "test_acc1": 5.926, "test_acc5": 16.886},
        {"epoch": 11, "train_loss": 6.399258077358478, "test_loss": 5.510222682516083, "test_acc1": 6.432, "test_acc5": 18.508},
        {"epoch": 12, "train_loss": 6.371843169616589, "test_loss": 5.454554444960966, "test_acc1": 7.418, "test_acc5": 20.028},
        {"epoch": 13, "train_loss": 6.331293023714098, "test_loss": 5.350319691286742, "test_acc1": 7.794, "test_acc5": 21.154},
        {"epoch": 14, "train_loss": 6.298067605660427, "test_loss": 5.270121920199794, "test_acc1": 8.692, "test_acc5": 22.93},
        {"epoch": 15, "train_loss": 6.279066535279055, "test_loss": 5.204319273242514, "test_acc1": 9.202, "test_acc5": 24.11},
        {"epoch": 16, "train_loss": 6.249703459590597, "test_loss": 5.125393623614129, "test_acc1": 10.53, "test_acc5": 26.338},
        {"epoch": 17, "train_loss": 6.21773943712869, "test_loss": 5.063503403700035, "test_acc1": 10.93, "test_acc5": 27.11},
        {"epoch": 18, "train_loss": 6.189450458769063, "test_loss": 5.014480741879412, "test_acc1": 11.504, "test_acc5": 28.228},
        {"epoch": 19, "train_loss": 6.160441656958318, "test_loss": 4.94392161150925, "test_acc1": 12.122, "test_acc5": 29.346},
        {"epoch": 20, "train_loss": 6.146470677968724, "test_loss": 4.9164143991834335, "test_acc1": 12.164, "test_acc5": 29.672},
        {"epoch": 21, "train_loss": 6.121253411015873, "test_loss": 4.852904168704084, "test_acc1": 13.108, "test_acc5": 30.738},
        {"epoch": 22, "train_loss": 6.115407601556431, "test_loss": 4.804972830619521, "test_acc1": 13.618, "test_acc5": 31.77},
        {"epoch": 23, "train_loss": 6.0937617498443775, "test_loss": 4.788146317460154, "test_acc1": 13.972, "test_acc5": 32.34},
        {"epoch": 24, "train_loss": 6.089793389788167, "test_loss": 4.755471413372127, "test_acc1": 14.246, "test_acc5": 32.892},
        {"epoch": 25, "train_loss": 6.0771706865604935, "test_loss": 4.738217310141061, "test_acc1": 14.504, "test_acc5": 33.29},
        {"epoch": 26, "train_loss": 6.06565475268342, "test_loss": 4.701416479722234, "test_acc1": 14.978, "test_acc5": 33.836},
        {"epoch": 27, "train_loss": 6.041946897257419, "test_loss": 4.683342833555382, "test_acc1": 15.07, "test_acc5": 34.25},
        {"epoch": 28, "train_loss": 6.040347081706928, "test_loss": 4.667666826539367, "test_acc1": 15.232, "test_acc5": 34.358},
        {"epoch": 29, "train_loss": 6.02230314570534, "test_loss": 4.65858539552179, "test_acc1": 15.418, "test_acc5": 34.746},
    ],
    "Fixed λ=0.9": [
        {"epoch": 0, "train_loss": 6.912791849954625, "test_loss": 6.895468431574698, "test_acc1": 0.184, "test_acc5": 0.7},
        {"epoch": 1, "train_loss": 6.905123210332138, "test_loss": 6.882541325256114, "test_acc1": 0.21, "test_acc5": 0.938},
        {"epoch": 2, "train_loss": 6.864318109792908, "test_loss": 6.714787606974594, "test_acc1": 0.584, "test_acc5": 2.538},
        {"epoch": 3, "train_loss": 6.803151724095347, "test_loss": 6.501723850046405, "test_acc1": 1.284, "test_acc5": 4.756},
        {"epoch": 4, "train_loss": 6.7207463947337205, "test_loss": 6.310936341758903, "test_acc1": 1.862, "test_acc5": 6.564},
        {"epoch": 5, "train_loss": 6.663979744923414, "test_loss": 6.158466466510569, "test_acc1": 2.34, "test_acc5": 8.152},
        {"epoch": 6, "train_loss": 6.60500994711152, "test_loss": 6.0062660770561855, "test_acc1": 3.092, "test_acc5": 10.224},
        {"epoch": 7, "train_loss": 6.554137958250921, "test_loss": 5.901272733702914, "test_acc1": 3.752, "test_acc5": 12.034},
        {"epoch": 8, "train_loss": 6.503946006878531, "test_loss": 5.777871688813653, "test_acc1": 4.434, "test_acc5": 14.112},
        {"epoch": 9, "train_loss": 6.46399299943221, "test_loss": 5.658197038956271, "test_acc1": 5.236, "test_acc5": 15.734},
        {"epoch": 10, "train_loss": 6.425742287931535, "test_loss": 5.563508496029686, "test_acc1": 6.118, "test_acc5": 17.42},
        {"epoch": 11, "train_loss": 6.383110873702854, "test_loss": 5.462640492970706, "test_acc1": 6.896, "test_acc5": 19.47},
        {"epoch": 12, "train_loss": 6.352921709532862, "test_loss": 5.400970473544288, "test_acc1": 7.834, "test_acc5": 20.87},
        {"epoch": 13, "train_loss": 6.309444859600996, "test_loss": 5.295211475314074, "test_acc1": 8.294, "test_acc5": 22.18},
        {"epoch": 14, "train_loss": 6.273598379871284, "test_loss": 5.207932774347204, "test_acc1": 9.256, "test_acc5": 24.07},
        {"epoch": 15, "train_loss": 6.253319106549253, "test_loss": 5.142084158103884, "test_acc1": 9.798, "test_acc5": 25.298},
        {"epoch": 16, "train_loss": 6.222549235130322, "test_loss": 5.059710522644393, "test_acc1": 10.898, "test_acc5": 27.366},
        {"epoch": 17, "train_loss": 6.189188841365656, "test_loss": 4.985927803825786, "test_acc1": 11.71, "test_acc5": 28.318},
        {"epoch": 18, "train_loss": 6.158809205313819, "test_loss": 4.925814546701562, "test_acc1": 12.326, "test_acc5": 29.948},
        {"epoch": 19, "train_loss": 6.128942475937257, "test_loss": 4.859368833876748, "test_acc1": 12.942, "test_acc5": 30.716},
        {"epoch": 20, "train_loss": 6.11440217806583, "test_loss": 4.823737235469673, "test_acc1": 13.192, "test_acc5": 31.304},
        {"epoch": 21, "train_loss": 6.087642549674124, "test_loss": 4.767980291643216, "test_acc1": 13.97, "test_acc5": 32.234},
        {"epoch": 22, "train_loss": 6.081580504706308, "test_loss": 4.715789987840726, "test_acc1": 14.46, "test_acc5": 33.308},
        {"epoch": 23, "train_loss": 6.0587789380568715, "test_loss": 4.700368719246551, "test_acc1": 14.722, "test_acc5": 33.762},
        {"epoch": 24, "train_loss": 6.054183290042613, "test_loss": 4.6621597905195395, "test_acc1": 15.144, "test_acc5": 34.49},
        {"epoch": 25, "train_loss": 6.040803900503122, "test_loss": 4.639180789467033, "test_acc1": 15.466, "test_acc5": 34.944},
        {"epoch": 26, "train_loss": 6.028794916515653, "test_loss": 4.60396116744471, "test_acc1": 16.004, "test_acc5": 35.668},
        {"epoch": 27, "train_loss": 6.003952831442328, "test_loss": 4.583701494085879, "test_acc1": 16.116, "test_acc5": 35.9},
        {"epoch": 28, "train_loss": 6.002018474420482, "test_loss": 4.570987417497708, "test_acc1": 16.292, "test_acc5": 36.174},
        {"epoch": 29, "train_loss": 5.982988643744003, "test_loss": 4.559197422202307, "test_acc1": 16.326, "test_acc5": 36.42},
    ],
}

colors = {"Learned λ": "#1f77b4", "Fixed λ=0.5": "#d62728", "Fixed λ=0.8": "#ff7f0e", "Fixed λ=0.9": "#2ca02c"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LION-D Tiny: Learned vs Fixed Decay λ (30 epochs, 500K ImageNet subset)", fontsize=13)

for label, epochs_data in data.items():
    epochs = [d["epoch"] for d in epochs_data]
    train_loss = [d["train_loss"] for d in epochs_data]
    val_acc1 = [d["test_acc1"] for d in epochs_data]
    val_acc5 = [d["test_acc5"] for d in epochs_data]
    c = colors[label]
    axes[0].plot(epochs, train_loss, label=label, color=c, linewidth=2)
    axes[1].plot(epochs, val_acc1, label=label, color=c, linewidth=2)
    axes[2].plot(epochs, val_acc5, label=label, color=c, linewidth=2)

titles = ["Training Loss", "Validation Top-1 Accuracy (%)", "Validation Top-5 Accuracy (%)"]
ylabels = ["Loss", "Top-1 (%)", "Top-5 (%)"]
for ax, title, ylabel in zip(axes, titles, ylabels):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/d/TUD/Q4/RS_SLS/results.png", dpi=150, bbox_inches="tight")
print("Saved to /mnt/d/TUD/Q4/RS_SLS/results.png")

print("\nFinal epoch (29) summary:")
print(f"{'Variant':<15} {'Top-1':>8} {'Top-5':>8} {'Train Loss':>12}")
print("-" * 46)
for label, epochs_data in data.items():
    d = epochs_data[-1]
    print(f"{label:<15} {d['test_acc1']:>8.2f}% {d['test_acc5']:>8.2f}% {d['train_loss']:>12.4f}")
