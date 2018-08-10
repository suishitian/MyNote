package main

import (
	//"bytes"
	"fmt"
	//"log"
	"net"
	//"os"
)

func CheckError(err error, str string) {
	if err != nil {
		fmt.Println(str)
	}
}

func handleClient(conn net.Conn) {
	for {
		send(conn, string(receive(conn)))
	}
}

func receive(conn net.Conn) {
	buffer := make([]byte, 1024)
	conn.Read(buffer)
	bufferr := buffer[5:]
	fmt.Println(string(bufferr))
}
func send(conn net.Conn, str string) {
	buffer := []byte(str)
	conn.Write(buffer)
	fmt.Println("send successfully")
}

func main() {

	fmt.Println("waiting for the client")
	netListen, err := net.Listen("tcp", "127.0.0.1:8888")
	defer func(listen net.Listener) {
		listen.Close()
	}(netListen)
	CheckError(err, "establish socket failed")

	for {
		conn, err := netListen.Accept()
		CheckError(err, "connected failed")
		go handleClient(conn)
	}
}
