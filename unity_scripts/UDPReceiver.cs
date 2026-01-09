using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;

public class UDPReceiver : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5005; // Πρέπει να είναι το ίδιο με την Python!

    // Εδώ αποθηκεύουμε την τελευταία θέση που λάβαμε
    private float targetX = 0f;

    // Ασφάλεια για threading (Unity API runs on main thread only)
    private object lockObject = new object();

    void Start()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("UDP Receiver Started...");
    }

    void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);

                // Το μήνυμα είναι "X,Y,Z". Το κόβουμε στα κόμματα.
                string[] coords = text.Split(',');

                if (coords.Length >= 1)
                {
                    // Διαβάζουμε το X (χρησιμοποιούμε InvariantCulture για την τελεία στα decimals)
                    float x = float.Parse(coords[0], CultureInfo.InvariantCulture);

                    lock (lockObject)
                    {
                        targetX = x; // Αποθήκευση για το Update()
                    }
                }
            }
            catch (System.Exception err)
            {
                Debug.Log(err.ToString());
            }
        }
    }

    void Update()
    {
        // Εδώ κουνάμε τον κύβο. Το Unity τρέχει το Update κάθε frame.
        float currentX;

        lock (lockObject)
        {
            currentX = targetX;
        }

        // Κουνάμε τον κύβο στον άξονα Χ (αριστερά-δεξιά)
        // Το divide by 2 είναι για να χωράει στην οθόνη
        transform.position = new Vector3(currentX / 2.0f, 0, 0);
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        client.Close();
    }
}