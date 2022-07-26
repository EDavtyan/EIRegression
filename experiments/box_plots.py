import matplotlib.pyplot as plt
import statistics


# Make a boxplot with an array of R2 and MAE results
plt.style.use("ggplot")
plt.rc('font', size=7)


def plotter():
    TITLE = "Concrete Dataset"

    # MAE and R2 from Random Forest, Gradient Boosting, Linear Regression, MLP Regressor and Fuzzy Regressor
    MAE_others = [[4.137786368714074, 4.463816585612668, 4.222136322088264, 4.4208405427083575, 4.000066018087506, 3.6026623848981054, 3.7848930253123036, 3.94659885163022, 3.966643523910271, 4.210883238869092, 4.008240842328969, 3.9272563898718285, 3.630914164149259, 4.03908293402897, 4.209443403546532, 4.56859779980653, 4.292426606427641, 4.04511165566991, 3.8533706980959255, 3.7904842408739374], [3.335608152454777, 3.4967316961978585, 3.770792627353266, 3.851754999077149, 3.3577802571674678, 3.29256569844346, 3.3737587763012167, 3.3396794075304497, 3.525587491232926, 3.441584875415279, 3.4793332428940564, 3.5490810489725613, 3.174855625692134, 3.6448546244001476, 3.5784603433001125, 3.773365035068288, 3.5679086111111125, 3.3501087301587265, 3.6021340448504997, 3.4249810760428194], [7.963627486800053, 8.903545763929227, 8.28108280226882, 8.342671249202459, 8.194618352066861, 7.792115202571787, 8.113698266338297, 8.38061787303978, 8.249281556800492,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            8.756071384925816, 8.545699591165505, 8.466137348873685, 8.36154262568049, 7.866983011724363, 8.269485101445182, 8.282009177402282, 8.57459175655063, 8.139670505763055, 7.922298868451542, 8.145845226511467], [3.54870751691744, 3.9976044885678896, 3.857587580652254, 4.455716509062597, 4.322334421814072, 4.039874085224712, 3.535222360933029, 3.3375555272774915, 3.7925748320693273, 3.702517209490538, 3.7221386751844916, 3.403381010837027, 3.076160463510557, 3.8308726017484274, 3.7161986563909903, 4.299296593121368, 4.141539705161291, 4.08442879452633, 4.436534951540727, 4.0845874902520105], [7.734882955242434, 8.69643451596326, 8.660281321091482, 7.809645546600912, 8.122642719572696, 8.437246201458748, 7.875987919571274, 8.15617063610238, 8.297811088557166, 7.976416783800331, 8.529888671070204, 8.304470729085981, 8.07523306125912, 7.75643727457849, 7.983112483393096, 8.48974107257465, 7.602572365491139, 8.282590748496766, 7.759748949470577, 7.692474105977124]]
    R2_others = [[0.8743296102164427, 0.871890794346428, 0.8819529118082552, 0.8591643443561869, 0.8980094374036095, 0.8982148739917657, 0.9044816292284484, 0.8813665886583536, 0.8918433749117591, 0.8711533558463525, 0.8983812755410054, 0.9035194054489929, 0.920127807925479, 0.8938069960355894, 0.8707945944537725, 0.8579011897685654, 0.8560551324367449, 0.8848425273623932, 0.876076849965067, 0.9047254098441172], [0.9106537341090852, 0.9145361459912086, 0.8949223308223782, 0.8756919089004708, 0.9179321748396255, 0.9079356645661272, 0.9167360435431604, 0.9084720134097367, 0.9025255856110325, 0.9096420418143821, 0.9199623931498925, 0.9157022579644927, 0.934109933246422, 0.9057688014561629, 0.898270653524593, 0.8910029890890301, 0.8948086014006789, 0.9179865353788418, 0.8988413291722634, 0.91621224439449], [0.6129611007434728, 0.5750464987485211, 0.6280209296531307, 0.5791068849868771, 0.6502851187950189, 0.5993488434912958, 0.6177718671790253, 0.5683224913488721, 0.6042688894287613, 0.5541833170148809,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0.6339801162740357, 0.6169413122427043, 0.6281589434759316, 0.6564209454866066, 0.5713239947612652, 0.6014545707776896, 0.5065663392807775, 0.6069567110894305, 0.6019204823428144, 0.6263629748353441], [0.8929701592747935, 0.878938923548932, 0.858680422407927, 0.7900898591691419, 0.8439877081856025, 0.8313812037216273, 0.8479139685433585, 0.8897192940320586, 0.8899967381371777, 0.8905530820400728, 0.9042861220597678, 0.9044703962524975, 0.9373220527582984, 0.853748985326109, 0.8174202844139966, 0.8298834869412688, 0.8259676660149854, 0.8061963944672662, 0.8468517984723803, 0.879071973671838], [0.6261806251566752, 0.5883200434988125, 0.5933583792681292, 0.6514584203899403, 0.6356093604769643, 0.5108720983184529, 0.6459735072740922, 0.4952498035149765, 0.5983047626578177, 0.616818452480846, 0.610243440926342, 0.6156479850728487, 0.6322792268605376, 0.6369320100518634, 0.5691877961928704, 0.5490183176726884, 0.6348766756775843, 0.6177899900353002, 0.5837553753976412, 0.6600162561900378]]
    # Boxes: Results obtained from Embedded Interpretable Regression
    EIREG_R2 = [0.8438158059455527, 0.8305681616853146,
                0.8742514778899529, 0.8434392695205205, 0.8427821287371646]
    EIREG_MAE = [4.3189494466639475, 4.574755023710729,
                 4.117929818005025, 4.2434082040974666, 4.437496139664071]

    # Boxes union
    boxes_R2 = [EIREG_R2] + R2_others
    boxes_MAE = [EIREG_MAE] + MAE_others

    # Title and labels
    labels = ["EI Regression", "GB", "RandomForest",
              "Linear", "MLP", "Fuzzy Rule"]

    fig, axes = plt.subplots(2, figsize=(5, 7), dpi=130)
    plt.subplots_adjust(left=0.08, bottom=None, right=1,
                        top=0.95, wspace=None, hspace=0.25)
    bpMAE = axes[0].boxplot(boxes_MAE, 0, '', labels=labels,
                            patch_artist=True, medianprops={'linewidth': 2, 'color': 'k'})
    axes[0].set_title(TITLE + r' $MAE$')

    bpMAE["boxes"][0].set(color='r', linewidth=2)
    bpMAE["boxes"][1].set(color='b', linewidth=2)
    bpMAE["boxes"][2].set(color='tab:orange', linewidth=2)
    bpMAE["boxes"][3].set(color='g', linewidth=2)
    bpMAE["boxes"][4].set(color='y', linewidth=2)
    bpMAE["boxes"][5].set(color='m', linewidth=2)

    bpR2 = axes[1].boxplot(boxes_R2, 0, '', labels=labels, patch_artist=True, medianprops={
        'linewidth': 2, 'color': 'k'})
    axes[1].set_title(TITLE + r' $R^2$')

    # Colors
    bpR2["boxes"][0].set(color='r', linewidth=2)
    bpR2["boxes"][1].set(color='b', linewidth=2)
    bpR2["boxes"][2].set(color='tab:orange', linewidth=2)
    bpR2["boxes"][3].set(color='g', linewidth=2)
    bpR2["boxes"][4].set(color='y', linewidth=2)
    bpR2["boxes"][5].set(color='m', linewidth=2)
    axes[1].set_ylim(0, 1)
    fig.savefig('examples/results/figures/' + TITLE+'.png')

    print("Embedded Interpretable Regression:\n")
    print("MAE")
    print("Mean: ", statistics.mean(EIREG_MAE))
    print("Std: ", statistics.stdev(EIREG_MAE),)

    print("R2")
    print("Mean: ", statistics.mean(EIREG_R2))
    print("Std: ", statistics.stdev(EIREG_R2), "\n")

    print("Gradient Boosting:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[0]))
    print("Std: ", statistics.stdev(MAE_others[0]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[0]))
    print("Std: ", statistics.stdev(R2_others[0]), "\n")

    print("Random Forest:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[1]))
    print("Std: ", statistics.stdev(MAE_others[1]),)

    print("R2")
    print("Mean: ", statistics.mean(R2_others[1]))
    print("Std: ", statistics.stdev(R2_others[1]), "\n")

    print("Linear:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[2]))
    print("Std: ", statistics.stdev(MAE_others[2]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[2]))
    print("Std: ", statistics.stdev(R2_others[2]), "\n")

    print("MLP:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[3]))
    print("Std: ", statistics.stdev(MAE_others[3]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[3]))
    print("Std: ", statistics.stdev(R2_others[3]), "\n")

    print("Fuzzy:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[4]))
    print("Std: ", statistics.stdev(MAE_others[4]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[4]))
    print("Std: ", statistics.stdev(R2_others[4]), "\n")
    plt.show()


plotter()
