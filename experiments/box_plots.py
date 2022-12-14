import matplotlib.pyplot as plt
import statistics


# Make a boxplot with an array of R2 and MAE results
plt.style.use("ggplot")
plt.rc("font", size=7)


def plotter():
    TITLE = "California Housing Dataset"

    # MAE and R2 from  Gradient Boosting, Random Forest, MLP Regressor, Linear Regression and Fuzzy Regressor
    MAE_others = [
        [
            38807.11410307156,
            41195.07909933636,
            40397.70509442616,
            38869.94849006721,
            39942.506876820255,
            40261.16996120162,
            40072.270174374105,
            39763.77768797037,
            38963.06417291837,
            41905.59233630108,
        ],
        [
            31775.986641472868,
            32324.076575581395,
            31464.310910852717,
            31206.242860465114,
            31551.60297286821,
            31397.396910852713,
            32038.825529069767,
            31787.888703488374,
            31955.455327519383,
            32141.4641744186,
        ],
        [
            37450.41153665312,
            38204.56603855018,
            37360.59450657931,
            36724.62542632916,
            38876.522813047064,
            37128.24380827077,
            38322.40396260891,
            37842.2317881506,
            37034.01813335603,
            38044.75797470743,
        ],
        [
            48973.409367956025,
            50854.965092921084,
            49926.55879187644,
            49129.021179826224,
            49550.79756816982,
            50076.08673859655,
            50106.71498474214,
            49377.985816230546,
            49387.18996258042,
            50592.00903684264,
        ],
        [
            49380.31908440523,
            50868.242584011175,
            49776.40121434415,
            49839.06942302291,
            50581.19244132632,
            48801.34028461668,
            49877.07574274308,
            50266.02926591261,
            49600.635777422474,
            48539.6713378281,
        ],
    ]

    R2_others = [
        [
            0.7461223348048773,
            0.7113180824775083,
            0.712395836773905,
            0.7424485667826421,
            0.7191393037263755,
            0.7165531284881579,
            0.7212621203090368,
            0.718307176811289,
            0.7310894727612705,
            0.6958948513573111,
        ],
        [
            0.8220100805415882,
            0.8181487618707914,
            0.8218552261193454,
            0.8253385021989705,
            0.8208273689769503,
            0.8235413369879382,
            0.8201236430606038,
            0.8172511232179036,
            0.8128560809187091,
            0.8138117681993363,
        ],
        [
            0.7796732135664833,
            0.7615302694216629,
            0.7557434379271307,
            0.7870137550317999,
            0.7442875492390153,
            0.7656079604708108,
            0.7608947674203563,
            0.7651262229272363,
            0.7713427232974772,
            0.7679998243765712,
        ],
        [
            0.6678149174947056,
            0.6285304400953036,
            0.6284767590986939,
            0.6662671257487482,
            0.6366730289738902,
            0.6358833926435462,
            0.644617283348748,
            0.6400279131540538,
            0.6453053372863876,
            0.6357418674974136,
        ],
        [
            0.6543595855314543,
            0.6475941177191649,
            0.6326419571306096,
            0.6413944405493619,
            0.6286885821410005,
            0.6531477107443492,
            0.6548709886816249,
            0.6337427891091696,
            0.6445708910382325,
            0.6613187195185493,
        ],
    ]

    # Boxes: Results obtained from Embedded Interpretable Regression
    EIREG_MAE = [
        [
            35171.77639248395,
            34412.28764968976,
            33949.427360379574,
            33475.585293359945,
            34029.177498021956,
            33877.11423173651,
            34394.238361557065,
            33169.79839901251,
            34652.621709089886,
            34444.07463391133,
            34480.31823377771,
            33988.159790142694,
            34321.36978993002,
            34081.8742589782,
            35586.34693044384,
        ],
        [
            32500.12188833431,
            31857.38016637307,
            31635.500785378747,
            30764.45015560775,
            32242.016531121553,
            32065.40000195733,
            32714.104639851244,
            31554.299328146404,
            32166.05260716383,
            31649.755119886475,
            32210.140845077316,
            31917.251254159328,
            32323.81061558035,
            31888.691781659814,
            32580.37057202975,
        ],
        [
            37874.41787883376,
            36975.97405881306,
            37457.047946731036,
            36844.683723808834,
            37605.95911004903,
            37353.5476094187,
            37460.40440613698,
            36945.68950494537,
            37524.98922266631,
            37464.137871798834,
            37550.51896174914,
            37725.48046911805,
            37898.94680152543,
            38026.30628044056,
            38494.4180966843,
        ],
        [
            46161.53414006252,
            45693.03605601019,
            45437.86736445492,
            44829.43605979177,
            45525.62045276088,
            45979.209116346414,
            45404.20489257042,
            44656.08934900147,
            45927.682036643135,
            46354.23525709088,
            45592.46124524881,
            45444.16385046376,
            45706.68895991267,
            45507.124978319065,
            46137.725669239124,
        ],
    ]

    EIREG_R2 = [
        [
            0.7713136715084608,
            0.7797282066152064,
            0.7864496242391261,
            0.7885257832404607,
            0.7768552680518191,
            0.790036366872261,
            0.7760496134805354,
            0.7885273783151495,
            0.7725107033984526,
            0.7846020382829597,
            0.7879214203216595,
            0.7831661303393445,
            0.7869222488611718,
            0.7861870907397207,
            0.770677525823239,
        ],
        [
            0.8138963761085091,
            0.8220745827511668,
            0.8263761814403241,
            0.8282363508048914,
            0.8078078461445137,
            0.8203133154368907,
            0.8059703677419572,
            0.8195178743939971,
            0.8128420065104536,
            0.8231259063628259,
            0.8220240551231459,
            0.8186291090149104,
            0.8214808863236913,
            0.8228268658314762,
            0.8143499207598626,
        ],
        [
            0.7705396858903619,
            0.770303506425529,
            0.7745448455339228,
            0.7727729737696472,
            0.7569600714127542,
            0.7595219403077481,
            0.7617149664317474,
            0.770945945341081,
            0.7647184967732401,
            0.7727567622982827,
            0.7734376927816431,
            0.7694244867066866,
            0.7717036631724602,
            0.7547379986071189,
            0.762598807554892,
        ],
        [
            0.6822761627753529,
            0.6873427812152708,
            0.6974793046716541,
            0.6939969381936961,
            0.6807888221419643,
            0.6868445967159953,
            0.6847668734782004,
            0.6949752736397388,
            0.6720275919734653,
            0.682365768292607,
            0.6954470391091776,
            0.6868471511334029,
            0.6943867416530648,
            0.6851898611786196,
            0.6869161443233649,
        ],
    ]

    # Boxes union
    boxes_MAE = (
        [MAE_others[0]]
        + [EIREG_MAE[0]]
        + [MAE_others[1]]
        + [EIREG_MAE[1]]
        + [MAE_others[2]]
        + [EIREG_MAE[2]]
        + [MAE_others[3]]
        + [EIREG_MAE[3]]
        + [MAE_others[4]]
    )
    boxes_R2 = (
        [R2_others[0]]
        + [EIREG_R2[0]]
        + [R2_others[1]]
        + [EIREG_R2[1]]
        + [R2_others[2]]
        + [EIREG_R2[2]]
        + [R2_others[3]]
        + [EIREG_R2[3]]
        + [R2_others[4]]
    )

    # Title and labels
    labels = [
        "GB",
        "GB+",
        "RF",
        "RF+",
        "MLP",
        "MLP+",
        "LR",
        "LR+",
        "FR",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=130)
    plt.subplots_adjust(left=0.05, right=0.97)
    bpMAE = axes[0].boxplot(
        boxes_MAE,
        0,
        "",
        labels=labels,
        patch_artist=True,
        medianprops={"linewidth": 2, "color": "k"},
    )
    axes[0].set_title(TITLE + r" $MAE$")

    bpMAE["boxes"][0].set(color="r", linewidth=2)
    bpMAE["boxes"][1].set(color="r", linewidth=2)
    bpMAE["boxes"][2].set(color="b", linewidth=2)
    bpMAE["boxes"][3].set(color="b", linewidth=2)
    bpMAE["boxes"][4].set(color="g", linewidth=2)
    bpMAE["boxes"][5].set(color="g", linewidth=2)
    bpMAE["boxes"][6].set(color="y", linewidth=2)
    bpMAE["boxes"][7].set(color="y", linewidth=2)
    bpMAE["boxes"][8].set(color="m", linewidth=2)

    bpR2 = axes[1].boxplot(
        boxes_R2,
        0,
        "",
        labels=labels,
        patch_artist=True,
        medianprops={"linewidth": 2, "color": "k"},
    )
    axes[1].set_title(TITLE + r" $R^2$")

    # Colors
    bpR2["boxes"][0].set(color="r", linewidth=2)
    bpR2["boxes"][1].set(color="r", linewidth=2)
    bpR2["boxes"][2].set(color="b", linewidth=2)
    bpR2["boxes"][3].set(color="b", linewidth=2)
    bpR2["boxes"][4].set(color="g", linewidth=2)
    bpR2["boxes"][5].set(color="g", linewidth=2)
    bpR2["boxes"][6].set(color="y", linewidth=2)
    bpR2["boxes"][7].set(color="y", linewidth=2)
    bpR2["boxes"][8].set(color="m", linewidth=2)
    axes[1].set_ylim(0, 1)
    fig.savefig("experiments/figures/" + TITLE + ".png")

    print("Gradient Boosting:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[0]))
    print("Std: ", statistics.stdev(MAE_others[0]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[0]))
    print("Std: ", statistics.stdev(R2_others[0]), "\n")

    print("Gradient Boosting Embedded:\n")
    print("MAE")
    print("Mean: ", statistics.mean(EIREG_MAE[0]))
    print("Std: ", statistics.stdev(EIREG_MAE[0]))

    print("R2")
    print("Mean: ", statistics.mean(EIREG_R2[0]))
    print("Std: ", statistics.stdev(EIREG_R2[0]), "\n")

    print("Random Forest:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[1]))
    print(
        "Std: ",
        statistics.stdev(MAE_others[1]),
    )

    print("R2")
    print("Mean: ", statistics.mean(R2_others[1]))
    print("Std: ", statistics.stdev(R2_others[1]), "\n")

    print("Random Forest Embedded:\n")
    print("MAE")
    print("Mean: ", statistics.mean(EIREG_MAE[1]))
    print(
        "Std: ",
        statistics.stdev(EIREG_MAE[1]),
    )

    print("R2")
    print("Mean: ", statistics.mean(EIREG_R2[1]))
    print("Std: ", statistics.stdev(EIREG_R2[1]), "\n")

    print("MLP:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[2]))
    print("Std: ", statistics.stdev(MAE_others[2]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[2]))
    print("Std: ", statistics.stdev(R2_others[2]), "\n")

    print("MLP Embedded:\n")
    print("MAE")
    print("Mean: ", statistics.mean(EIREG_MAE[2]))
    print(
        "Std: ",
        statistics.stdev(EIREG_MAE[2]),
    )

    print("R2")
    print("Mean: ", statistics.mean(EIREG_R2[2]))
    print("Std: ", statistics.stdev(EIREG_R2[2]), "\n")

    print("Linear:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[3]))
    print("Std: ", statistics.stdev(MAE_others[3]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[3]))
    print("Std: ", statistics.stdev(R2_others[3]), "\n")

    print("Linear Embedded:\n")
    print("MAE")
    print("Mean: ", statistics.mean(EIREG_MAE[3]))
    print(
        "Std: ",
        statistics.stdev(EIREG_MAE[3]),
    )

    print("R2")
    print("Mean: ", statistics.mean(EIREG_R2[3]))
    print("Std: ", statistics.stdev(EIREG_R2[3]), "\n")

    print("Fuzzy:\n")
    print("MAE")
    print("Mean: ", statistics.mean(MAE_others[4]))
    print("Std: ", statistics.stdev(MAE_others[4]))

    print("R2")
    print("Mean: ", statistics.mean(R2_others[4]))
    print("Std: ", statistics.stdev(R2_others[4]), "\n")
    plt.show()


plotter()


def weave(list1, list2):
    lijst = []
    i = 0
    while i <= len(list1):
        lijst += [list1[i]]
        lijst += [list2[i]]
        i + 1
    return lijst
