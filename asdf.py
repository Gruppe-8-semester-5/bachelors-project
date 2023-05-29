accuracies = [
    0.19864285714285715,
    0.129,
    0.14185714285714285,
    0.19657142857142856,
    0.1955,
    0.19707142857142856,
    0.17214285714285715,
    0.1775,
    0.16792857142857143,
    0.128,
    0.1975,
    0.33485714285714285,
    0.17,
    0.33664285714285713,
    0.4655,
    0.30807142857142855,
    0.38492857142857145,
    0.29528571428571426,
    0.27785714285714286,
    0.2760714285714286,
    0.2102857142857143,
    0.32257142857142856,
    0.3018571428571429,
    0.289,
    0.42114285714285715,
    0.39207142857142857,
    0.2869285714285714,
    0.5037857142857143,
    0.31892857142857145,
    0.3272857142857143,
    0.23542857142857143,
    0.4552857142857143,
    0.39671428571428574,
    0.4197142857142857,
    0.47828571428571426,
    0.45235714285714285,
    0.3919285714285714,
    0.5467142857142857,
    0.3469285714285714,
    0.4975,
    0.21635714285714286,
    0.3380714285714286,
    0.2737142857142857,
    0.5567142857142857,
    0.4432857142857143,
    0.5719285714285715,
    0.5607142857142857,
    0.3315714285714286,
    0.45007142857142857,
    0.47735714285714287,
    0.18714285714285714,
    0.20792857142857143,
    0.26914285714285713,
    0.5050714285714286,
    0.49614285714285716,
    0.5177857142857143,
    0.5123571428571428,
    0.5080714285714286,
    0.49692857142857144,
    0.4722142857142857,
    0.1352142857142857,
    0.30207142857142855,
    0.3655,
    0.5133571428571428,
    0.49328571428571427,
    0.6428571428571429,
    0.5880714285714286,
    0.6812142857142857,
    0.5642857142857143,
    0.3187142857142857,
    0.19528571428571428,
    0.2892857142857143,
    0.40114285714285713,
    0.39385714285714285,
    0.5926428571428571,
    0.5739285714285715,
    0.6877142857142857,
    0.6720714285714285,
    0.6103571428571428,
    0.5849285714285715,
    0.2025,
    0.23735714285714285,
    0.4509285714285714,
    0.36828571428571427,
    0.4918571428571429,
    0.45985714285714285,
    0.6429285714285714,
    0.663,
    0.671,
    0.5887857142857142,
    0.18764285714285714,
    0.229,
    0.3028571428571429,
    0.4742142857142857,
    0.3955714285714286,
    0.5163571428571428,
    0.5933571428571428,
    0.6108571428571429,
    0.6502142857142857,
    0.6397142857142857,
    0.20007142857142857,
    0.28485714285714286,
    0.2765714285714286,
    0.3302857142857143,
    0.4085,
    0.5027857142857143,
    0.4584285714285714,
    0.45021428571428573,
    0.6435714285714286,
    0.7281428571428571,
]

best = 0
best_i = 0
for i, x in enumerate(accuracies):
    if x > best:
        best = x
        best_i = i

layers = [
    " NN accuracy (10,10) - ",
    "NN accuracy (10,20) - ",
    "NN accuracy (10,30) - ",
    "NN accuracy (10,40) - ",
    "NN accuracy (10,50) - ",
    "NN accuracy (10,70) - ",
    "NN accuracy (10,90) - ",
    "NN accuracy (10,120) -",
    "NN accuracy (10,200) -",
    "NN accuracy (10,300) -",
    "NN accuracy (20,10) - ",
    "NN accuracy (20,20) - ",
    "NN accuracy (20,30) - ",
    "NN accuracy (20,40) - ",
    "NN accuracy (20,50) - ",
    "NN accuracy (20,70) - ",
    "NN accuracy (20,90) - ",
    "NN accuracy (20,120) -",
    "NN accuracy (20,200) -",
    "NN accuracy (20,300) -",
    "NN accuracy (30,10) - ",
    "NN accuracy (30,20) - ",
    "NN accuracy (30,30) - ",
    "NN accuracy (30,40) - ",
    "NN accuracy (30,50) - ",
    "NN accuracy (30,70) - ",
    "NN accuracy (30,90) - ",
    "NN accuracy (30,120) -",
    "NN accuracy (30,200) -",
    "NN accuracy (30,300) -",
    "NN accuracy (40,10) - ",
    "NN accuracy (40,20) - ",
    "NN accuracy (40,30) - ",
    "NN accuracy (40,40) - ",
    "NN accuracy (40,50) - ",
    "NN accuracy (40,70) - ",
    "NN accuracy (40,90) - ",
    "NN accuracy (40,120) -",
    "NN accuracy (40,200) -",
    "NN accuracy (40,300) -",
    "NN accuracy (50,10) - ",
    "NN accuracy (50,20) - ",
    "NN accuracy (50,30) - ",
    "NN accuracy (50,40) - ",
    "NN accuracy (50,50) - ",
    "NN accuracy (50,70) - ",
    "NN accuracy (50,90) - ",
    "NN accuracy (50,120) -",
    "NN accuracy (50,200) -",
    "NN accuracy (50,300) -",
    "NN accuracy (70,10) - ",
    "NN accuracy (70,20) - ",
    "NN accuracy (70,30) - ",
    "NN accuracy (70,40) - ",
    "NN accuracy (70,50) - ",
    "NN accuracy (70,70) - ",
    "NN accuracy (70,90) - ",
    "NN accuracy (70,120) -",
    "NN accuracy (70,200) -",
    "NN accuracy (70,300) -",
    "NN accuracy (90,10) - ",
    "NN accuracy (90,20) - ",
    "NN accuracy (90,30) - ",
    "NN accuracy (90,40) - ",
    "NN accuracy (90,50) - ",
    "NN accuracy (90,70) - ",
    "NN accuracy (90,90) - ",
    "NN accuracy (90,120) -",
    "NN accuracy (90,200) -",
    "NN accuracy (90,300) -",
    "NN accuracy (120,10) -",
    "NN accuracy (120,20) -",
    "NN accuracy (120,30) -",
    "NN accuracy (120,40) -",
    "NN accuracy (120,50) -",
    "NN accuracy (120,70) -",
    "NN accuracy (120,90) -",
    "NN accuracy (120,120) ",
    "NN accuracy (120,200) ",
    "NN accuracy (120,300) ",
    "NN accuracy (160,10) -",
    "NN accuracy (160,20) -",
    "NN accuracy (160,30) -",
    "NN accuracy (160,40) -",
    "NN accuracy (160,50) -",
    "NN accuracy (160,70) -",
    "NN accuracy (160,90) -",
    "NN accuracy (160,120) ",
    "NN accuracy (160,200) ",
    "NN accuracy (160,300) ",
    "NN accuracy (200,10) -",
    "NN accuracy (200,20) -",
    "NN accuracy (200,30) -",
    "NN accuracy (200,40) -",
    "NN accuracy (200,50) -",
    "NN accuracy (200,70) -",
    "NN accuracy (200,90) -",
    "NN accuracy (200,120) ",
    "NN accuracy (200,200) ",
    "NN accuracy (200,300) ",
    "NN accuracy (300,10) -",
    "NN accuracy (300,20) -",
    "NN accuracy (300,30) -",
    "NN accuracy (300,40) -",
    "NN accuracy (300,50) -",
    "NN accuracy (300,70) -",
    "NN accuracy (300,90) -",
    "NN accuracy (300,120) ",
    "NN accuracy (300,200) ",
    "NN accuracy (300,300) ",
]


print(layers[best_i])
