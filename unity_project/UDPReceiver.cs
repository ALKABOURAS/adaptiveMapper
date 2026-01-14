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
    public int port = 5005;

    // Μεταβλητές για τις θέσεις X και Y
    private float targetX = 0f;
    private float targetY = 0f;

    // Ασφάλεια για threading
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
                // --- Η ΓΡΑΜΜΗ ΠΟΥ ΕΛΕΙΠΕ ---
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP); // Εδώ ορίζεται το 'data'
                // ---------------------------

                string text = Encoding.UTF8.GetString(data);

                // Το Python στέλνει "X,Y,Z" -> Κόβουμε στα κόμματα
                string[] coords = text.Split(',');

                // Ελέγχουμε αν λάβαμε τουλάχιστον 2 τιμές (X και Y)
                if (coords.Length >= 2)
                {
                    // Parse με InvariantCulture για να καταλαβαίνει την τελεία (.) ως υποδιαστολή
                    float x = float.Parse(coords[0], CultureInfo.InvariantCulture);
                    float y = float.Parse(coords[1], CultureInfo.InvariantCulture);

                    lock (lockObject)
                    {
                        targetX = x;
                        targetY = y;
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
        float curX, curY;

        lock (lockObject)
        {
            curX = targetX;
            curY = targetY;
        }

        // Κίνηση σε X και Y.
        // Διαιρούμε με 2.0f για να χωράει η κίνηση στην οθόνη του Unity
        transform.position = new Vector3(curX / 2.0f, curY / 2.0f, 0);
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (client != null) client.Close();
    }
}