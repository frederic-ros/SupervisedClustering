# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 12:36:27 2025

@author: frederic.ros
"""
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_method_times(folder):
    """
    Parcourt un dossier et extrait les temps des méthodes contenues dans 
    les fichiers de type <num>_d*.txt, puis trace un graphique des temps.
    
    Paramètre
    ---------
    folder : str
        Chemin vers le dossier contenant les fichiers texte.
    """
    times = defaultdict(list)

    # ✅ Recherche des fichiers du type "1_d2.txt", "2_d5.txt", etc.
    files = [f for f in os.listdir(folder) if re.match(r"\d+_d.*\.txt$", f)]

    # ✅ Tri basé sur le numéro avant le "_"
    files = sorted(files, key=lambda f: int(f.split('_')[0]))

    pattern = re.compile(r"(\w+)\(time,M,R,S\):\s*([\d.]+)")

    for file in files:
        with open(os.path.join(folder, file), 'r') as f:
            content = f.read()
            for match in pattern.finditer(content):
                method = match.group(1)
                value = float(match.group(2))
                times[method].append(value)

    if not times:
        print("Aucune donnée trouvée. Vérifiez le dossier et le format des fichiers.")
        return

    # ✅ Graphique plus haut pour mieux voir les courbes
    plt.figure(figsize=(10, 8))
    x = range(len(files))

    for method, values in sorted(times.items()):
        plt.plot(x, values, marker='o', label=method)

    plt.title("Computation Time")
    plt.ylabel("Time(s)")

    # ✅ Abscisses sans .txt et sans numéro de préfixe
    xlabels = [f.split('_')[1].replace('.txt', '') for f in files]
    plt.xticks(ticks=x, labels=xlabels)

    # ✅ Légende compacte, police réduite, plus proche du graphe
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,                # plus de colonnes = moins de lignes
        #fontsize='x-small',    # légende plus discrète
        frameon=False
    )

    # ✅ Ajuste les marges pour garder de la place au graphique
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


plot_method_times("../time")