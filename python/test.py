import socket

HOST='165.229.125.122'
PORT= 8000

client_socket= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
client_socket.sendall('하이염'.encode())

client_socket.close()
