# Plan du projet : Asking Under Attack

Plan de travail du 9 novembre 2026 au 14 avril 2027. Il découle du README (§12 à §15 et §19). En cas de conflit, c'est le README qui compte.

---

## 1. Vue d'ensemble

| Phase | Semaines | Dates | Objectif | GPU |
|---|---|---|---|---|
| P0 : mise en place | S1 | 9 – 15 nov. | Dépôt, papier qui compile, positionnement corrigé | 0 h |
| P1 : banc d'essai | S2 – S4 | 16 nov. – 6 déc. | AgentDojo-Ask codé et testé sur CPU, baselines, prereg | ~10 h |
| P2 : E1 + décision | S5 – S6 | 7 – 20 déc. | E1 (RQ0) puis **point de décision le 20 déc.** | ~25 h |
| P3 : entraînement | S7 – S12 | 21 déc. – 31 janv. | FA-GRPO-L (principal + ablations), brouillon du preprint | ~120 h |
| P4 : attaques + preprint | S13 – S14 | 1 – 14 févr. | E4, E5, E6. **Preprint arXiv** | ~35 h |
| P5 : version ECAI | S15 – S18 | 15 févr. – 14 mars | Papier de 7 pages, annexes, code public | 0 h |
| P6 : marge | S19 – S22 | 15 mars – 11 avril | Relances, relecture, dernière recherche de nouveauté | 0 à 20 h |
| — | — | **14 avril 2027** | **Soumission ECAI** | — |

**Règle clé (stage)** : le code doit être **terminé avant le début du stage** (entre janvier et mars). Il faut donc finir P1 et la préparation de P3 au plus tard début janvier. Ensuite, les entraînements tournent seuls en mode commit sur Kaggle.

**Quota GPU** : 30 h par semaine sur un seul compte. Les ~190 h prévues tiennent dans les semaines S4 à S14, avec un peu de marge.

---

## 2. Détail par phase

### P0 : mise en place (S1, 9 – 15 nov.)

| # | Tâche | Résultat attendu |
|---|---|---|
| 0.1 | Décompresser `asking_under_attack_sources.zip` (`paper/`, `sim/`), `git init`, créer `CLAUDE.md` (`@README.md`), créer l'arborescence du §19.1 | Dépôt propre, premier commit |
| 0.2 | Lancer `sim/simulate.py --fast`, puis la version complète, puis compiler `paper/main.tex` | Les chiffres du Tableau 2 sont reproduits |
| 0.3 | Corrections 1 à 4 du §17.2 : contributions (GRPO), étude pilote passée en travail futur, Tableau 1 (Sha / ARLAS / Basu), Related Work | Le papier recompile, positionnement à jour |
| 0.4 | Vérifier les références ajoutées (pas d'invention ; sinon `TODO`) | `refs.bib` à jour |
| 0.5 | Vérifier Ancker 2017 : la baisse de « 30 % » est-elle bien un rapport de cotes de 0,7 ? | Forme Ancker validée ou corrigée |

**Livrable** : papier corrigé + dépôt Git.

### P1 : banc d'essai AgentDojo-Ask (S2 – S4, 16 nov. – 6 déc.)

| Semaine | Tâches | Tests |
|---|---|---|
| S2 (16 – 22 nov.) | Installer AgentDojo. Lancer une tâche banking avec un faux modèle. Lire l'API : tâches, ground truth, utilité, sécurité, `BaseAttack`, pipeline. Noter tout dans `docs/notes.md`. Coder `ada/users.py` | Profils (linéaire, exponentiel, Ancker, marche), décisions, mise à jour de la charge $\ell$ |
| S3 (23 – 29 nov.) | `ada/ask.py` (dry-run, effet résolu, hash, refus si l'effet change). `ada/sessions.py` (version simple : remise à zéro de l'environnement par tâche, charge partagée). `ada/metrics.py` | Une approbation ne passe pas sur un autre effet. La charge passe d'une tâche à l'autre. Les mesures sont correctes sur des logs factices |
| S4 (30 nov. – 6 déc.) | `ada/attacks.py` (inondation, suppression, déguisement). Baselines B1 à B5, B3+ et B8. **`docs/prereg.md`** (P1 à P3, critère du 20 déc.). Premier petit run GPU (Qwen3-1.7B, quelques sessions) pour mesurer les temps | Toutes les baselines tournent sur CPU avec un faux modèle |

**Livrables** : `ada/` testé, `baselines/`, `prereg.md`, une estimation réelle du coût GPU d'une session.

**À vérifier tôt** : la part des tâches que Qwen3-1.7B réussit **sans attaque**. Si elle est trop basse, on garde seulement les tâches réussies, ou on commence par banking et slack (§15.2).

### P2 : E1 et point de décision (S5 – S6, 7 – 20 déc.)

| Tâche | Détail |
|---|---|
| Notebook Kaggle E1 | vLLM + Qwen3-1.7B. Petit run, puis run complet |
| Conditions | Baselines B1 à B5, B3+, B8 × {humain parfait, humain fatigué} × {sans inondation, avec inondation} |
| Analyse | Classement des politiques, τ de Kendall, intervalles bootstrap, effondrement de B2 |
| En parallèle (CPU) | Préparer l'entraînement : choisir ART ou TRL, **vérifier la normalisation des avantages** (`scale_rewards`), fp16, Qwen3 sans thinking. Run court de test |

**Point de décision du 20 décembre**

| Résultat de E1 | Suite |
|---|---|
| La fatigue change nettement le classement | **Continuer** vers P3 |
| Elle ne le change pas | Version atelier (banc d'essai + analyse) ou arrêt |

Noter la décision dans le §22 du README.

### P3 : entraînement FA-GRPO-L (S7 – S12, 21 déc. – 31 janv.)

Ordre des runs, du plus important au moins important :

| # | Run | Rôle | Graines |
|---|---|---|---|
| 1 | FA-GRPO-L complet, 1.7B | Méthode principale (RQ1, RQ2) | 3 si possible |
| 2 | B7 : RL sans fatigue ($r\equiv1$) | Ablation 1, baseline clé | 2 |
| 3 | Sans compteur de charge | Ablation 2 | 1 |
| 4 | Pénalité fixe à la place de la contrainte | Ablation 3 | 1 |
| 5 | Un seul profil | Ablation 4 | 1 |
| 6 | Échelle des avantages par groupe ou scalarisée | Ablation 7 | 1 |
| 7 | Sans attaques sur l'humain à l'entraînement | Ablation 6 | 1 |

- Un run de ~15 h = deux sessions en mode commit de 9 h, avec reprise depuis un checkpoint.
- Cadence : environ 2 runs par semaine (30 h).
- Pendant les runs : écrire le brouillon du preprint (méthode, banc d'essai, E1).
- Les ablations 5 et 8 (§11) ne demandent pas d'entraînement : elles se font en évaluation seulement.

**Si l'entraînement ne bat pas B5 ni B3+** : papier « banc d'essai + analyse » pour un atelier (§15.2).

### P4 : attaques, généralisation, preprint (S13 – S14, 1 – 14 févr.)

| ID | Contenu | GPU |
|---|---|---|
| E4 (RQ3) | Injections statiques, AutoDojo, inondation (on fait varier $k$), suppression, déguisement, attaquant adaptatif. ASR dans le pire cas | ~20 h |
| E5 (RQ4) | Profils jamais vus, suite travel, passage à 4B | ~15 h |
| E6 (RQ2) | Demandes selon la position dans la session, régression logistique de ASK sur (risque, charge), comparaison avec $\ell^\dagger$ | ~0 h |

- Tableaux et figures produits **par des scripts** à partir de `results/`.
- Refaire la recherche de nouveauté (mots-clés du §4).
- **Preprint arXiv vers le 14 février.**

### P5 : version ECAI (S15 – S18, 15 févr. – 14 mars)

- Réduire le papier de 20 pages à environ 7 pages + 1 page de références. Preuves et détails en annexe.
- Vérifier les règles d'ECAI 2027 (pages, double aveugle, preprints, date du résumé).
- Rendre le code public (version anonymisée pour la relecture).
- Mettre à jour la section éthique (pas d'étude humaine).

### P6 : marge (S19 – S22, 15 mars – 11 avril)

- Relancer les runs ratés, ajouter des graines si le quota le permet.
- Relecture complète, vérification de chaque chiffre (script → tableau → texte).
- Dernière recherche de nouveauté.
- Soumettre le résumé (vérifier la date), puis le papier le **14 avril**.

---

## 3. Jalons

| Date | Jalon | Critère |
|---|---|---|
| 15 nov. | Dépôt prêt | Le papier compile, corrections 1 à 4 faites |
| 6 déc. | Banc d'essai prêt | Tests verts, baselines OK sur CPU, prereg écrit |
| **20 déc.** | **Décision GO / NO GO** | Le classement change avec la fatigue |
| 10 janv. | Code figé | Entraînement lancé en mode commit, avant le stage |
| 31 janv. | Entraînements finis | Run principal + ablations prioritaires |
| 14 févr. | Preprint arXiv | Résultats E1 à E6 |
| 14 mars | Version ECAI | 7 pages + annexes |
| 14 avril | Soumission ECAI | — |

---

## 4. Risques principaux

| Risque | Signal | Réponse |
|---|---|---|
| Un papier concurrent sort | Recherche arXiv régulière | Se concentrer sur l'axe C, ou se présenter comme une extension |
| Le modèle 1.7B réussit trop peu de tâches | Mesure en S4 | Filtrer les tâches, ou commencer par banking et slack |
| Le RL ne tient pas sur T4 | Run de test en S6 | Réduire $K$, $G$ et la longueur des réponses |
| La normalisation par groupe casse $\lambda$ | $\lambda$ explose (cf. 174 dans la simulation) | Échelle au niveau du batch, vérifiée avant P3 |
| Le quota GPU ne suffit pas | Coût mesuré en S4 | Couper les graines et les ablations secondaires, et le dire dans le papier |
| Le stage commence tôt | Date connue | Figer le code dès le 10 janvier |
| Instabilité tardive de l'entraînement | Bascule de la politique | Checkpoints fréquents, amortissement PID du pas dual |
