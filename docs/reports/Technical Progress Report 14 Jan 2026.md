# 📄 Τεχνική Αναφορά Προόδου: Adaptive Motion Navigation

**Θέμα:** Μελέτη και αξιολόγηση τεχνικών πλοήγησης σε τρισδιάστατα περιβάλλοντα με χρήση motion-based controllers και adaptive mapping.  
**Ημερομηνία Αναφοράς:** Ιανουάριος 2026  
**Στάδιο:** Υλοποίηση Πρωτοτύπου & Tuning (Prototype & Tuning Phase)

---

## 1. Περίληψη & Στόχος

Σκοπός της εργασίας είναι η γεφύρωση του χάσματος (research gap) μεταξύ VR motion controllers και Desktop 3D περιβαλλόντων. Αναπτύσσουμε ένα σύστημα που επιτρέπει τη χρήση Joy-Con/Pro Controller ως συσκευή κατάδειξης (pointing device) με ακρίβεια ποντικιού, χρησιμοποιώντας προηγμένους αλγορίθμους φιλτραρίσματος (Kalman) και δυναμικής προσαρμογής ευαισθησίας (Adaptive Mapping).

---

## 2. Αρχιτεκτονική Συστήματος (System Architecture)

Το σύστημα υλοποιήθηκε με βάση την εξής ροή δεδομένων (Pipeline):

1. **Hardware Layer:** Nintendo Joy-Con (L/R) ή Pro Controller (Bluetooth HID).
    
2. **Driver Layer (Python):** Custom driver με χρήση βιβλιοθήκης hidapi για λήψη Raw IMU Data (Gyroscope/Accelerometer) στα 66Hz.
    
3. **Processing Layer:**
    
    - **Calibration:** Αυτόματη αφαίρεση σταθερού σφάλματος (Bias removal / ZUPT).
        
    - **Conversion:** Μετατροπή Raw values σε Degrees Per Second (dps).
        
    - **Filtering:** Εφαρμογή 1D Kalman Filter ανά άξονα για μείωση θορύβου (Jitter).
        
    - **Adaptive Mapping:** Μη-γραμμική αντιστοίχιση ταχύτητας εισόδου σε ταχύτητα κέρσορα (Precision Mode vs Fast Mode).
        
4. **Communication Layer:** UDP Socket (Localhost:5005) για αποστολή δεδομένων σε πραγματικό χρόνο.
    
5. **Application Layer (Unity):** 3D Testbed για οπτικοποίηση και εκτέλεση πειραμάτων (Fitts’ Law tasks).
    

---

## 3. Χρονολόγιο Υλοποίησης (Implementation Log)

### Φάση 1: Προσομοίωση & Θεμελίωση (Simulation Phase)

Λόγω αρχικής έλλειψης hardware, αναπτύχθηκε ένα πλήρως λειτουργικό περιβάλλον εξομοίωσης.

- **Mock IMU:** Δημιουργία Python script (mock_imu.py) που παράγει τεχνητά δεδομένα κίνησης με προσθήκη Gaussian Noise.
    
- **Kalman Filter:** Υλοποίηση κλάσης SimpleKalmanFilter. Αρχικές δοκιμές έδειξαν επιτυχή εξομάλυνση του θορύβου.
    
- **Adaptive Logic:** Υλοποίηση του AdaptiveMapper.
    
    - Λογική: Αν velocity < threshold, τότε gain = gain * precision_factor.
        
    - Αποτέλεσμα: Επιτυχής μετάβαση σε "Sniper Mode" σε χαμηλές ταχύτητες.
        

### Φάση 2: Διασύνδεση με Unity (The Bridge)

- Δημιουργία UDP Server (Python) και UDP Receiver (C# Unity).
    
- Επιτυχής μεταφορά συντεταγμένων σε πραγματικό χρόνο.
    
- Δημιουργία 3D σκηνής με στόχους (Targets) και μετρητή χρόνου (Fitts’ Law logic).
    
- Ενσωμάτωση επιλογής "Mouse Control" στο Unity για τη δημιουργία Baseline δεδομένων (σύγκριση με ποντίκι).
    

### Φάση 3: Ενσωμάτωση Hardware (Hardware Integration)

- **Επίλυση Προβλημάτων Driver:**
    
    - Αντιμετώπιση προβλημάτων με βιβλιοθήκες hid vs hidapi στα Windows (DLL conflicts).
        
    - Αντικατάσταση του 8BitDo adapter με απλό Bluetooth Dongle (TP-Link UB500) για πρόσβαση σε Raw Data.
        
- **Reverse Engineering Joy-Con:**
    
    - Αποστολή εντολών αφύπνισης (0x01, 0x40) για ενεργοποίηση του 6-Axis Sensor.
        
    - Ενεργοποίηση High Performance Mode (Input Report 0x30).
        
    - Parsing των raw bytes σε signed integers.
        

### Φάση 4: Βελτιστοποίηση & Calibration (Optimization)

- **Drift Removal:** Το γυροσκόπιο παρουσίαζε ολίσθηση (drift).
    
    - Λύση: Υλοποίηση **Auto-Calibration (ZUPT)**. Ο αλγόριθμος ανιχνεύει πότε το χειριστήριο είναι ακίνητο (βάσει variance buffer) και επαναϋπολογίζει το Bias.
        
- **Pro Controller Support:** Επέκταση του driver για υποστήριξη και του Pro Controller (Product ID 0x2009).
    
- **Mapping Correction:** Διόρθωση των αξόνων για φυσική αίσθηση:
    
    - Pitch (Πάνω/Κάτω) -> Gyro X.
        
    - Yaw (Αριστερά/Δεξιά) -> Gyro Z.
        
    - Invert Y-Axis option.
        

### Φάση 5: Tuning (Η Μάχη Lag vs Stability)

Διεξήχθησαν πειράματα για τη βελτιστοποίηση της αίσθησης χειρισμού.

- **Πρόβλημα:** Παρατηρήθηκε αίσθηση "Lag" στις απότομες κινήσεις.
    
- **Διάγνωση:** Το πρόβλημα δεν ήταν καθυστέρηση δικτύου, αλλά υπερβολικά χαμηλό Sensitivity (SENSITIVITY=100) σε συνδυασμό με βαρύ φίλτρο Kalman (R=10).
    
- **Τελική Ρύθμιση (Gaming Config):**
    
    - Measurement Noise (R): **2.0** (Ισορροπία ομαλότητας/ταχύτητας).
        
    - Sensitivity: **25.0 - 30.0** (Ταχύτερη απόκριση).
        
    - Adaptive Threshold: **8.0** (Ενεργοποίηση Precision Mode μόνο σε πολύ αργές κινήσεις).
        
    - Loop Rate: Μείωση του time.sleep στο **0.001s**.
        

---

## 4. Τρέχουσα Κατάσταση Κώδικα (Key Code Components)

### A. Python Driver (Core Logic)

Ο driver υποστηρίζει πλέον:

1. Αναγνώριση Joy-Con L/R και Pro Controller.
    
2. Ανάγνωση Gyroscope (dps) και Analog Sticks.
    
3. Αυτόματο καλιμπράρισμα (Auto-Zeroing).
    

### B. Unity Testbed

Το περιβάλλον στο Unity περιλαμβάνει:

1. Κύβο που ελέγχεται μέσω UDP.
    
2. Σύστημα στόχων που εμφανίζονται τυχαία.
    
3. Καταγραφή χρόνου απόκρισης (Time to Hit).
    

---

## 5. Επόμενα Βήματα (Next Steps)

1. **Interaction Implementation:** Υλοποίηση "Click" (π.χ. μέσω Spacebar ή κουμπιού του Joy-Con) για την ολοκλήρωση του task.
    
2. **Data Logging:** Αυτόματη καταγραφή των δεδομένων του πειράματος σε CSV (Time, Distance, Errors) για στατιστική ανάλυση.
    
3. **User Study:** Εκτέλεση του πειράματος με χρήστες συγκρίνοντας:
    
    - Mouse (Baseline)
        
    - Joy-Con (Raw)
        
    - Joy-Con (Filtered + Adaptive - Η μέθοδός μας)