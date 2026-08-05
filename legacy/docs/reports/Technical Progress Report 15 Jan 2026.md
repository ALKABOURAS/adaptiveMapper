# 📄 Technical Report #2: Advanced Filtering & Dynamic Interaction
**Date:** 14 Ιανουαρίου 2026
**Status:** ✅ Milestone Reached (Dynamic Precision Logic)
**Tags:** #thesis #filtering #one-euro #interaction-design

---

## 1. Σύνοψη (Executive Summary)
Σε αυτή τη φάση της έρευνας, μεταβήκαμε από τη στατική επεξεργασία σήματος (Simple Kalman) σε **προσαρμοστικά φίλτρα (OneEuro Filter)** και υλοποιήσαμε μηχανισμούς δυναμικής αλληλεπίδρασης. Το σύστημα πλέον δεν αντιδρά μόνο στην κίνηση, αλλά και στην πρόθεση του χρήστη (μέσω Hardware Triggers), αλλάζοντας τη συμπεριφορά του σε πραγματικό χρόνο ("Sniper Mode" vs "Fast Mode"). Επίσης, επιλύθηκαν κρίσιμα ζητήματα απόδοσης (Lag) και ομαλότητας (Stuttering).

---

## 2. Τεχνικές Αναβαθμίσεις (Key Implementations)

### A. Μετάβαση στο 1€ Filter (One Euro Filter)
Αντικαταστήσαμε τον Kalman Filter με τον **1€ Filter**, ο οποίος σχεδιάστηκε ειδικά για HCI.
*   **Το Πρόβλημα του Kalman:** Με σταθερό `R` (Noise Covariance), έπρεπε να διαλέξουμε ή *smoothness* (με lag) ή *speed* (με jitter).
*   **Η Λύση του 1€:** Προσαρμόζει τη συχνότητα αποκοπής (`cutoff`) βάσει της ταχύτητας του χεριού.
    *   *Ακίνητο:* Cutoff $\approx$ 0.1 (Απόλυτη σταθερότητα).
    *   *Γρήγορο:* Cutoff $\approx$ >1.0 (Μηδενικό Latency).

### B. Hardware Trigger Integration
Ενημερώθηκε ο `joycon_driver.py` ώστε να διαβάζει την κατάσταση των κουμπιών (**ZL / ZR**) μέσα στο ίδιο πακέτο δεδομένων (HID Report) με το γυροσκόπιο.
*   **Αποτέλεσμα:** Μηδενική καθυστέρηση στην ανίχνευση πατήματος, χωρίς ανάγκη για δεύτερο USB call.

### C. Υλοποίηση "Sniper Mode" (Dynamic Tuning)
Αντί για απλή μείωση της ευαισθησίας (που προκαλούσε phase-out/stuttering σε υψηλές τιμές διαιρέτη), υλοποιήσαμε **Δυναμική Αλλαγή Παραμέτρων**:

| Κατάσταση | Sensitivity | Filter Beta | Min Cutoff | Velocity Clamp |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (Released)** | High (Fast) | 6.0 (Agile) | 1.0 (Light) | Off |
| **Sniper (Held)** | Low (Precise) | 0.5 (Heavy) | 0.01 (Rock solid) | **ON (Max 0.1)** |

> [!SUCCESS] The Clamping Breakthrough
> Η εισαγωγή του **Velocity Clamping** (`clamp(step, -0.1, 0.1)`) έλυσε το πρόβλημα των απότομων κινήσεων στο Precision Mode. Ακόμα και αν ο χρήστης κάνει απότομη κίνηση 2 μέτρων, ο κώδικας επιτρέπει μέγιστη μετατόπιση 0.1 μονάδας, διατηρώντας την αίσθηση του "βαρύ" χειριστηρίου.

### D. Εξάλειψη του Lag (Performance Fix)
Εντοπίστηκε ότι η εντολή `time.sleep(0.001)` στα Windows προκαλούσε καθυστέρηση ~15ms ανά frame λόγω του OS Scheduler resolution.
*   **Fix:** Αφαίρεση του sleep. Το loop πλέον τρέχει στο μέγιστο ρυθμό που επιτρέπει το USB polling (blocking read), εξαλείφοντας το input lag.

---

## 3. Αναδιοργάνωση Project (Code Structure)

Ο φάκελος `tests` καθαρίστηκε και οργανώθηκε σε υπο-φακέλους για σαφήνεια:

```bash
tests/
├── legacy_simulation/   # Παλιά tests με mock data (Kalman basics)
├── utils/               # Εργαλεία debugging (console print, plotters)
├── experiments/         # ΤΑ ΕΝΕΡΓΑ SCRIPTS
│   ├── main_navigation.py      # Το τελικό script πλοήγησης
│   ├── main_nav_precision.py   # Το script με το Sniper Mode logic
│   └── live_filter_comparison.py
└── graphs/              # Αποθηκευμένα γραφήματα απόδοσης
```

## 4. Επόμενα Βήματα (Next Steps)

**Comparative Plotting (Real-Time):**  
- Δημιουργία ενός νέου run_comparison_plot.py που θα τρέχει ταυτόχρονα τον Kalman και τον 1€ Filter (με τις νέες δυναμικές ρυθμίσεις) για να οπτικοποιήσουμε τη διαφορά στην απόκριση "Flick" και στο "Micro-adjustment".  
Στόχος: Να μπει ως εικόνα στο κεφάλαιο "Filter Evaluation" της διπλωματικής.

- **Unity Interaction (Clicking):**  
Χαρτογράφηση του πατήματος ZL/ZR (ή άλλου κουμπιού) σε εντολή "Fire/Click" στο Unity για να ολοκληρωθεί το Fitts' Law task.

- **Data Logging Pipeline:**  
Αυτόματη καταγραφή (CSV) του χρόνου που χρειάζεται ο χρήστης για να πετύχει τον στόχο (Target Acquisition Time) και του Error Rate, για τη στατιστική ανάλυση.

> [!NOTE] Research Note  
> Η αίσθηση του "Sniper Mode" με το clamping θυμίζει έντονα επαγγελματικά συστήματα CAD ή high-end gaming mice. Αυτό είναι σημαντικό εύρημα για τη χρηστικότητα σε Desktop περιβάλλοντα.

