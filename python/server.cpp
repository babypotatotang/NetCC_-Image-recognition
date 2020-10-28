#pragma comment(lib, "ws2_32.lib")
#include <WinSock2.h>
#include <stdio.h>
#include <WS2tcpip.h>
#define MAX_BUF 1024

int main(void) {

	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
		printf("Error !!");
		return -1;
	}

	SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
	if (s == INVALID_SOCKET) {
		printf("Error in socket(), Error code: %d\n", WSAGetLastError());
		WSACleanup();
		return -1;
	}

	SOCKADDR_IN myAddress;
	ZeroMemory(&myAddress, sizeof(myAddress));
	myAddress.sin_family = AF_INET;
	myAddress.sin_port = htons(50000);
	myAddress.sin_addr.s_addr = htonl(INADDR_ANY);

	if (bind(s, (SOCKADDR*)&myAddress, sizeof(myAddress)) == SOCKET_ERROR) {
		printf("Error in bind(), Error code: %d\n", WSAGetLastError());
		closesocket(s);
		WSACleanup();
		return -1;
	}

	listen(s, 5);


	while (1) {
		SOCKADDR_IN clientAddress;
		int iAddressLength = sizeof(clientAddress);

		SOCKET t = accept(s, (SOCKADDR*)&clientAddress, &iAddressLength);

		char chRxBuf[MAX_BUF] = {};
		recv(t, chRxBuf, MAX_BUF, 0);
		printf("Recv Msg : %s\n", chRxBuf);

		const char chTxBuf[MAX_BUF] = "Hi, Client. Current time is...";
		send(t, chTxBuf, strlen(chTxBuf), 0);
		printf("Send Msg : %s\n", chTxBuf);

		closesocket(t);
	}

	closesocket(s);
	WSACleanup();

	return 0; // 정상 종료
}