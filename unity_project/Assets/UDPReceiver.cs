using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;

public class UDPReceiver : MonoBehaviour
{
    // --- ΕΠΙΛΟΓΗ MODES ---
    [Header("Input Settings")]
    public bool useMouseControl = false; // Αν είναι TRUE, αγνοεί το Python και ακούει το ποντίκι
    public float mouseSensitivity = 0.1f;

    // --- UDP Variables ---
    Thread receiveThread;
    UdpClient client;
    public int port = 5005;
    private float targetX = 0f;
    private float targetY = 0f;
    private object lockObject = new object();
    private bool isRunning = true; // Flag για σωστό κλείσιμο του thread

    void Start()
    {
        // Ξεκινάμε το UDP thread πάντα (μπορεί να θέλουμε να γυρίσουμε mode real-time)
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData()
    {
        try {
            client = new UdpClient(port);
            while (isRunning)
            {
                try
                {
                    IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                    // Το Receive μπλοκάρει, οπότε αν κλείσουμε το app θέλει προσοχή
                    byte[] data = client.Receive(ref anyIP);
                    
                    string text = Encoding.UTF8.GetString(data);
                    string[] coords = text.Split(',');

                    if (coords.Length >= 2)
                    {
                        float x = float.Parse(coords[0], CultureInfo.InvariantCulture);
                        float y = float.Parse(coords[1], CultureInfo.InvariantCulture);

                        lock (lockObject)
                        {
                            targetX = x;
                            targetY = y;
                        }
                    }
                }
                catch (System.Exception) 
                { 
                    // Ignored usually (socket close)
                }
            }
        }
        catch (System.Exception e) { Debug.Log(e.ToString()); }
    }

    void Update()
    {
        // ΕΔΩ ΕΙΝΑΙ Η ΑΛΛΑΓΗ: Επιλέγουμε πηγή εισόδου
        if (useMouseControl)
        {
            MoveWithMouse();
        }
        else
        {
            MoveWithUDP();
        }
    }

    void MoveWithUDP()
    {
        float curX, curY;
        lock (lockObject)
        {
            curX = targetX;
            curY = targetY;
        }
        // Κίνηση βάσει Python
        transform.position = new Vector3(curX / 2.0f, curY / 2.0f, 0);
    }

    void MoveWithMouse()
    {
        // Μετατροπή θέσης ποντικιού (Screen Pixels) σε World Coordinates
        Vector3 mousePos = Input.mousePosition;
        
        // Το Z είναι η απόσταση της κάμερας από τον κύβο (η κάμερα είναι στο -10, ο κύβος στο 0 -> απόσταση 10)
        mousePos.z = 10.0f; 
        
        Vector3 worldPos = Camera.main.ScreenToWorldPoint(mousePos);
        
        // Εφαρμογή θέσης
        transform.position = worldPos;
    }

    void OnApplicationQuit()
    {
        isRunning = false;
        if (receiveThread != null) receiveThread.Abort();
        if (client != null) client.Close();
    }
}