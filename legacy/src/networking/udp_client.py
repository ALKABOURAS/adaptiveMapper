import socket
import struct

class UDPSender:
    def __init__(self, ip="127.0.0.1", port=5005):
        """
        ip: 127.0.0.1 σημαίνει 'ο εαυτός μου' (localhost).
        port: Μια 'πόρτα' για να ακούει το Unity (π.χ. 5005).
        """
        self.udp_ip = ip
        self.udp_port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_data(self, val_x, val_y, val_z):
        """
        Στέλνει 3 τιμές (X, Y, Z) στο Unity.
        Τις μετατρέπουμε σε string μορφής "X,Y,Z" γιατί είναι εύκολο να διαβαστούν.
        """
        message = f"{val_x:.4f},{val_y:.4f},{val_z:.4f}"

        # Στέλνουμε το μήνυμα encoded σε bytes
        self.sock.sendto(message.encode('utf-8'), (self.udp_ip, self.udp_port))
        # print(f"Sent: {message}") # Ξε-σχολίασε για debug