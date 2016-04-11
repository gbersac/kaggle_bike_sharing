# Test techniques quantmetry

## Partie I – Statistiques descriptives

Tout d'abord il est important de modifier le jeu de données afin de pouvoir exploiter la colonne `datetime`. Celle-ci va être découplée en date ("yyyy/mm/dd") => `date`, heure => `hours` et jour de l'année => `dayOfYear`.

Le principal outil d'exploration de données sera la matrice de dispertion d'une variable en fonction d'une autre. Comme nous cherchons à prédire la variable `count` (dernière ligne, dernière collone), nous regardons ses dépendances. Nous désignerons un graphique de la matrice a partir des index (0, 0) désignant le graphique en haut à gauche.

- Les corrélations les plus évidentes avec `count` sont `registered`, `casual`, `hour` et `weather`.
- Les corrélations moins fortes mais significatives avec `count` sont `workingday`, `dayOfYear` et `holiday`.
- Les autres semblent avoir moins d'importance.
