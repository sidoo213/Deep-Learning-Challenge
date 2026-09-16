# TSM — Temporal Shift Module

Documentation détaillée du modèle implémenté dans `tsm.py`, conforme à Track A (entraîné from scratch).

**Référence** : Lin, Gan, Han. *"TSM: Temporal Shift Module for Efficient Video Understanding"*, ICCV 2019.

---

## 1. L'idée centrale

Le problème en reconnaissance d'action vidéo : un CNN 2D est aveugle au temps (il voit chaque frame indépendamment), et un CNN 3D ajoute beaucoup de paramètres et de FLOPs (la convolution 3D coûte ~7× plus cher qu'une 2D équivalente).

L'idée de TSM : **donner une capacité temporelle à un CNN 2D classique pour zéro paramètre supplémentaire et un coût FLOPs négligeable**, en *déplaçant* une petite fraction des canaux dans le temps avant chaque convolution.

Pourquoi ça marche : si avant la convolution on déplace 1/8 des canaux d'une frame vers la frame suivante (et 1/8 dans l'autre sens), alors chaque frame voit un mini-aperçu du passé et du futur dans ses propres canaux. La convolution 2D suivante peut donc mélanger ces canaux "passé/présent/futur" et apprendre des features temporelles, sans aucune conv 3D.

C'est en gros "une conv 3D pour les pauvres" — mais en pratique, ça atteint des performances comparables aux modèles 3D sur Something-Something.

---

## 2. L'opération de shift en détail (`_temporal_shift`)

```python
def _temporal_shift(x: torch.Tensor, num_segments: int, fold_div: int = 8) -> torch.Tensor:
```

### Entrée et sortie

- **Entrée** `x` de shape `(B*T, C, H, W)` — toutes les frames du batch concaténées le long de la dimension batch. C'est le format dans lequel un CNN 2D les traite naturellement.
- **Sortie** : tenseur de même shape, avec une fraction des canaux décalés dans le temps.

### Étapes

1. Reshape de `(B*T, C, H, W)` vers `(B, T, C, H, W)` pour exposer la dimension temporelle.
2. Calcul de `fold = C // fold_div`. Avec `fold_div=8` (défaut), `fold` représente 1/8 des canaux.
3. Création d'un tenseur de sortie `out` initialisé à zéros, même shape.
4. Trois opérations en parallèle sur les canaux :

| Canaux | Action | Code |
|---|---|---|
| `[0 : fold]` (les premiers 1/8) | **shift gauche** : frame t reçoit ces canaux depuis frame t+1 | `out[:, :-1, :fold] = x[:, 1:, :fold]` |
| `[fold : 2*fold]` (les 1/8 suivants) | **shift droite** : frame t reçoit ces canaux depuis frame t-1 | `out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]` |
| `[2*fold : C]` (les 6/8 restants) | **non touchés** : restent identiques | `out[:, :, 2*fold:] = x[:, :, 2*fold:]` |

5. Reshape inverse vers `(B*T, C, H, W)`.

### Visualisation

Imagine `T=4` frames et `C=8` canaux (`fold_div=8` donc `fold=1`). Avant shift, le tenseur d'une vidéo ressemble à :

```
frame 0:  [c0_0, c0_1, c0_2, c0_3, c0_4, c0_5, c0_6, c0_7]
frame 1:  [c1_0, c1_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7]
frame 2:  [c2_0, c2_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7]
frame 3:  [c3_0, c3_1, c3_2, c3_3, c3_4, c3_5, c3_6, c3_7]
```

Après shift (`fold=1`) :

```
frame 0:  [c1_0, 0,    c0_2, c0_3, c0_4, c0_5, c0_6, c0_7]  ← canal 0 vient du futur
frame 1:  [c2_0, c0_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7]  ← canal 0 du futur, canal 1 du passé
frame 2:  [c3_0, c1_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7]
frame 3:  [0,    c2_1, c3_2, c3_3, c3_4, c3_5, c3_6, c3_7]  ← canal 0 perdu (pas de futur)
```

Donc à frame 1 par exemple, sa représentation contient :
- canal 0 = info de frame 2 (vision du futur)
- canal 1 = info de frame 0 (vision du passé)
- canaux 2-7 = ses propres infos

La conv 2D suivante mélange ces 8 canaux ensemble → produit naturellement des features temporelles.

### Bord et fuite d'information

Les zones où `out[:, :-1, ...]` et `out[:, 1:, ...]` n'écrivent rien restent à zéro :
- Frame 0 perd le slot du "shift droite" (pas de frame -1) → ce canal vaut 0
- Frame T-1 perd le slot du "shift gauche" (pas de frame T) → ce canal vaut 0

C'est volontaire et symétrique. Le modèle apprend à compenser ces zéros aux extrémités.

---

## 3. Le bloc résiduel TSM (`_TSMResidualBlock`)

Architecture standard ResNet18 basic block (deux conv 3×3 + skip), avec **une seule modification** : le temporal shift est appliqué à l'entrée de la première conv.

```
x (B*T, C_in, H, W)
├── shortcut (identity ou 1×1 conv si stride/channels changent)
└── shift → conv1 (3×3) → BN → ReLU → conv2 (3×3) → BN
                                                       │
                                       (identity) + ──┘
                                            │
                                          ReLU
```

Note importante : **seule la première conv reçoit l'entrée shiftée**. Le skip est calculé sur le `x` non-shifté (l'identité). Cela évite de propager le décalage temporel hors du bloc — chaque bloc fait son propre mélange temporel sur sa première conv, puis "remet à jour" la trajectoire par le skip.

Si le shift était appliqué sur le skip aussi, on accumulerait les décalages couche après couche, ce qui finirait par tout brouiller.

---

## 4. L'architecture globale (`TSM`)

Layout ResNet18 standard :

| Stage | Operation | Output shape (frame) | Output channels |
|---|---|---|---|
| stem | conv 7×7 stride 2 + BN + ReLU + maxpool 3×3 stride 2 | 56×56 | 64 |
| layer1 | 2 × `_TSMResidualBlock(64, 64)` | 56×56 | 64 |
| layer2 | `_TSMResidualBlock(64, 128, stride=2)` + 1 × `_TSMResidualBlock(128, 128)` | 28×28 | 128 |
| layer3 | `_TSMResidualBlock(128, 256, stride=2)` + 1 × `_TSMResidualBlock(256, 256)` | 14×14 | 256 |
| layer4 | `_TSMResidualBlock(256, 512, stride=2)` + 1 × `_TSMResidualBlock(512, 512)` | 7×7 | 512 |
| pool | global average pool 2D | 1×1 | 512 |

Soit **8 blocs TSM au total**, donc 8 opérations de shift temporel intercalées dans le réseau. Chaque shift opère sur un niveau d'abstraction différent : les premiers shifts mélangent des features bas-niveau (bords, textures), les derniers mélangent des features sémantiques (parties d'objet, structure spatiale).

### Tête de classification

```
GAP → (B*T, 512)
↓
reshape vers (B, T, 512)
↓
mean pool sur T → (B, 512)         # temporal mean pool final
↓
dropout(0.5)
↓
Linear(512, num_classes)
```

L'agrégation temporelle finale est un simple **mean pool sur les T descripteurs de frames**. Combiné avec les 8 shifts intermédiaires, on a déjà beaucoup mélangé les frames — le mean pool joue le rôle d'une moyenne pondérée *implicite* du raisonnement temporel.

### Initialisation

Le modèle est entraîné from scratch (Track A). L'init est :
- Conv2d : **Kaiming normal** (fan_out, ReLU) — adapté aux activations ReLU
- BatchNorm2d : weight=1, bias=0 — identité au démarrage
- Linear : **truncated normal** std=0.02 — petite échelle, évite saturation softmax

---

## 5. Forward pass complet

```python
def forward(self, video: torch.Tensor) -> torch.Tensor:
    B, T, C, H, W = video.shape
    assert T == self.num_segments    # T doit matcher

    x = video.reshape(B * T, C, H, W)   # frames stackées dans le batch

    x = self.stem(x)                    # 224×224 → 56×56, 64 canaux
    x = self.layer1(x)                  # 2 blocs TSM
    x = self.layer2(x)                  # 56→28, +2 blocs TSM
    x = self.layer3(x)                  # 28→14, +2 blocs TSM
    x = self.layer4(x)                  # 14→7,  +2 blocs TSM

    x = self.gap(x).flatten(1)          # (B*T, 512)
    x = x.view(B, T, -1).mean(dim=1)    # (B, 512)
    x = self.dropout(x)
    return self.classifier(x)           # (B, num_classes)
```

Coût computationnel : essentiellement celui d'un ResNet18 2D appliqué `T` fois (les shifts ajoutent ~0% en FLOPs car ils sont juste des copies de slices). Donc bien moins cher qu'un 3D-ResNet équivalent.

---

## 6. Hyperparamètres du config

| Hyperparam | Défaut | Effet |
|---|---|---|
| `num_segments` | 8 | Nombre de frames par clip. **Doit être égal à `dataset.num_frames`** (assert dans `forward`). |
| `base_channels` | 64 | Canaux du stem. La dernière couche a `base_channels * 8` canaux (512 par défaut). |
| `dropout` | 0.5 | Régularisation sur la sortie du temporal mean pool, juste avant le classifieur. |
| `fold_div` | 8 | `1/fold_div` des canaux sont shiftés vers l'avant, idem vers l'arrière. Donc 25% des canaux sont "touchés" à chaque bloc avec la valeur par défaut. |
| `pretrained` | false | Accepté pour compatibilité API mais ignoré (Track A : from scratch). |

### Choix typiques

- `fold_div=8` : valeur du papier original. `fold_div=4` (50% des canaux shiftés) augmente le signal temporel mais peut détruire des features spatiales utiles.
- `num_segments=8` : standard sur SSv2. Plus = plus de temps mais plus de compute.
- `dropout=0.5` : agressif, justifié pour un modèle from-scratch sur peu de données.

---

## 7. Configuration d'entraînement (`configs/experiment/tsm.yaml`)

| Param | Valeur | Pourquoi |
|---|---|---|
| `epochs` | 50 | TSM from-scratch a besoin de beaucoup d'epochs |
| `batch_size` | 24 | 24 × 8 frames = 192 frames/step, gérable sur 24 Go VRAM |
| `optimizer` | sgd | Recette canonique du papier TSM |
| `lr` | 0.01 | Couplé à SGD + momentum + weight_decay |
| `momentum` | 0.9 | Standard Nesterov-style |
| `weight_decay` | 5e-4 | Régularisation L2 modérée |
| `use_horizontal_flip` | **false** | **CRITIQUE** sur Something-Something — voir ci-dessous |
| `use_random_crop` | true (scale 0.6-1.0) | Augmentation spatiale recommandée |
| `use_color_jitter` | true (0.2) | Augmentation chromatique modérée |
| `use_amp` | true | Mixed precision pour la vitesse |
| `num_workers` | 4 | Conservateur, à monter si le GPU stalle |
| `persistent_workers` | true | Évite le respawn de workers entre epochs |

### Pourquoi `use_horizontal_flip: false` est non-négociable

Something-Something contient des classes **directionnelles** (`Pulling something from left to right`, `Pulling something from right to left`, etc.). Un flip horizontal **transforme une classe en son inverse**. Si tu actives le flip, ton modèle apprend à confondre systématiquement ces classes — ton accuracy plafonne mécaniquement.

C'est le piège #1 sur Something-Something. Notre experiment yaml le désactive explicitement.

---

## 8. Comment lancer

```bash
uv run python src/train.py experiment=tsm \
  training.checkpoint_path=best_tsm.pt training.device=cuda
```

Ou avec overrides custom :

```bash
uv run python src/train.py experiment=tsm \
  training.epochs=30 training.batch_size=32 training.lr=0.02 \
  training.checkpoint_path=best_tsm.pt training.device=cuda
```

Évaluation et submission inchangées (le squelette `evaluate.py` / `create_submission.py` rebuildra TSM via `build_model`) :

```bash
uv run python src/evaluate.py training.checkpoint_path=best_tsm.pt
uv run python src/create_submission.py training.checkpoint_path=best_tsm.pt
```

---

## 9. Modifications appliquées au squelette du code

Pour que TSM soit utilisable comme n'importe quel autre modèle, les changements suivants ont été faits dans le squelette :

1. **`src/train.py`** :
   - Import ajouté : `from models.tsm import TSM`
   - Branche ajoutée dans `build_model()` : `if name == "tsm": return TSM(...)`
   - Branche ajoutée dans la sélection d'optimiseur : `elif optimizer_name == "sgd": ...` avec support de `momentum` et `nesterov` (optionnels dans la config)

2. **`src/configs/model/tsm.yaml`** : créé avec le préambule Hydra standard (`# @package _global_` + wrapper `model:`).
3. **`src/configs/experiment/tsm.yaml`** : preset d'entraînement (LR=0.01, SGD, 50 epochs, augmentation SSv2-safe).

`evaluate.py` et `create_submission.py` ne sont pas modifiés — ils rebuildront TSM automatiquement via le `config` sauvegardé dans le checkpoint.

---

## 10. Performance attendue et positionnement

| Modèle (Track A, from scratch) | Top-1 attendu (val) | Coût |
|---|---|---|
| `cnn_baseline` | 5-15 % | très léger |
| `cnn_lstm` | 10-20 % | léger |
| `cnn_transformer` | 15-25 % | moyen |
| `two_stream_transformer` | 15-25 % | lourd |
| **`tsm` (50 epochs)** | **25-40 %** | moyen (ResNet18 × T) |

TSM est généralement le meilleur modèle Track A sur Something-Something parmi les architectures "from scratch" raisonnables. C'est la baseline standard de la littérature SSv2 pour les modèles 2D efficaces.

Pour aller au-delà, il faut passer à Track B (transformers pré-entraînés) ou ajouter des techniques avancées (TTA, ensemble, etc. — voir `openworld.md` et `problems.md`).

---

## 11. Empirical findings — best known config (ablations, val 6745 clips, 2026-05-24)

**These are measured results, not theory. When generating a TSM training command, use the parameters below by default.** Each row notes whether the evidence is TSM-specific (clean A/B holding everything else fixed) or carried over from another model family.

| Factor | Config key | **Use** | Evidence |
|---|---|---|---|
| Class-balanced sampler | `training.use_class_balanced_sampler` | **false** | TSM-specific: `samp0_cos_tconv` 0.3204 > `samp1_cos_tconv` 0.3088. Test≈train distribution → rebalancing trades top-1 for tail recall. |
| LR scheduler | `training.scheduler` / `training.warmup_epochs` | **cosine, warmup 3** | TSM-specific: cos > flat at both poolings (+1.6 to +2.1 pt). Biggest single lever. |
| Temporal pooling | `model.temporal_pool` | **mean** | TSM-specific: mean > tconv (cos: 0.3294 vs 0.3088; flat: 0.3084 vs 0.2928). The learned `tconv` overfits the tiny distinct-frame signal. |
| Optimizer | `training.optimizer` | **adamw** | The strong seed9010 TSM runs used AdamW. SGD-vs-AdamW was not A/B'd on TSM directly, but SGD collapsed to ~0.15 on the CNN baseline. Pair AdamW with `lr≈1e-3` (≈10× below the SGD 0.01 in `tsm.yaml`). |
| Dropout | `model.dropout` | **0.5** | Cross-model (CNN baseline): drop0.5 ≈0.318 > drop0.1 ≈0.307. Aggressive dropout helps from-scratch. |
| Label smoothing | `training.label_smoothing` | **0.1 (or 0.0)** | Wash — no top-1 signal in any pairing. Keep 0.1 for head calibration. |
| Horizontal flip | `training.use_horizontal_flip` | **false** | Non-negotiable on SSv2 (directional classes — see §7). |

**Best single TSM observed:** `samp1_cos_mean` = 0.3294 top-1 (≈ tied with the best CNN baseline at 0.3291). The per-factor winners point at **nocbs + cosine + mean pooling**; the `samp0_cos_mean` combination was not run and is the obvious next experiment.

**Recommended command (best known TSM):**

```bash
uv run python src/train.py experiment=tsm \
  training.optimizer=adamw training.lr=1e-3 \
  training.scheduler=cosine training.warmup_epochs=3 \
  training.use_class_balanced_sampler=false \
  model.temporal_pool=mean model.dropout=0.5 \
  training.label_smoothing=0.1 \
  training.use_horizontal_flip=false \
  training.checkpoint_path=tsm_best.pt training.device=cuda
```

**The real lever is the ensemble**, not any single model: blending diverse architectures reached **0.398 top-1 / 0.728 top-5** (vs ~0.33 for the best single model). Keep architecturally diverse members (even weaker ones decorrelate errors), exclude diverged runs (two collapsed to 0.0821), and fit ensemble weights on **OOF** predictions (`oof_predict.py`) so the gain is leakage-free.
