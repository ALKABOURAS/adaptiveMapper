using UnityEngine;

public class CursorCollision : MonoBehaviour
{
    public ExperimentManager manager;

    void Start()
    {
        // ΑΥΤΟΜΑΤΗ ΕΥΡΕΣΗ:
        // Αν ξεχάσαμε να το βάλουμε στο Inspector, το βρίσκει μόνο του.
        if (manager == null)
        {
            manager = FindObjectOfType<ExperimentManager>();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            // Έλεγχος ασφαλείας: Υπάρχει ο manager;
            if (manager != null)
            {
                manager.TargetHit();
            }
            else
            {
                Debug.LogError("Δεν βρέθηκε το ExperimentManager script στη σκηνή!");
            }
        }
    }
}