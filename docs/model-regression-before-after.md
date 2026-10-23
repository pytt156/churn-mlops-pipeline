# Före och efter: Model regression gate

Körning (UTC): 2026-10-23T08:27:32+00:00

## Problem

Den tidigare kvalitetsgrinden kontrollerade bara ROC AUC >= 0,70.
En modell kunde därmed vara sämre än den godkända modellen och ändå passera.

## Metod

Testkandidaten tränades på samma data och samma train/test-uppdelning
som Champion, med `LogisticRegression(C=1e-6)` i samma förbehandlingspipeline.

Detta är en avsiktligt svagare demonstrationsmodell, inte en föreslagen release.
Kandidatens modellfil och metadata skapades i en temporär katalog, utan att
ändra `outputs/` eller `reference/`.

Båda kvalitetsgrindarna kördes på kandidatens verkligt beräknade mätvärden.

## Resultat

| Kontroll | Resultat |
|---|---|
| Champion ROC AUC | 0.788671 |
| Challenger ROC AUC | 0.779927 |
| Skillnad | -0.008745 |
| Före: absolutgrind >= 0,70 | Godkänd (exit 0) |
| Efter: Champion–Challenger | Stoppad |

Referenscommit: `8887b1112306901294694f8853d389217ff28790`
Data SHA256: `51df885d247889baa04dce12ed4fc8b55faccefe75b00afa3342e2d731c481bb`

## Slutsats och begränsning

Den nya grinden stoppar denna avsiktligt försämrade kandidat, som den gamla
grinden släppte igenom.

Detta demonstrerar kontrolllogik, inte en statistisk garanti för förbättrad
generalisering. Testuppsättningen används också av träningsflödet, vilket
begränsar användbarheten som oberoende slututvärdering.
