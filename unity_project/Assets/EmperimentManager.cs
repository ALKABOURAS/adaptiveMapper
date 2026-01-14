using UnityEngine;
using System.Collections;

public class ExperimentManager : MonoBehaviour
{
    public GameObject target; // Reference to the target sphere
    public GameObject cursor; // Reference to our controlled cube
    
    // Bounds for random position (Adjust based on your screen/camera)
    public float xRange = 8.0f; 
    public float yRange = 4.0f;

    private float startTime;
    private int hitCount = 0;

    void Start()
    {
        if (target == null)
            Debug.LogError("Target is not assigned in Inspector!");
            
        MoveTarget(); // Initial spawn
    }

    public void TargetHit()
    {
        // 1. Calculate time taken
        float duration = Time.time - startTime;
        hitCount++;

        // 2. Log data (For now, just console)
        Debug.Log($"Hit #{hitCount} | Time: {duration:F3} seconds");

        // 3. Move target to new random position
        MoveTarget();
    }

    void MoveTarget()
    {
        // Generate random X, Y inside the screen bounds
        float randomX = Random.Range(-xRange, xRange);
        float randomY = Random.Range(-yRange, yRange);

        target.transform.position = new Vector3(randomX, randomY, 0);
        
        // Reset timer
        startTime = Time.time;
    }
}