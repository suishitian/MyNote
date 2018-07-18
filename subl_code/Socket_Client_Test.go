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

func receiveInfo(conn net.Conn) {
	for {
		buffer := make([]byte, 1024)
		conn.Read(buffer)
		fmt.Println("sendBack: " + string(buffer))
	}
}

func send(conn net.Conn, str string) {
	buffer := []byte(str)
	conn.Write(buffer)
	fmt.Println("send successfully")
}
func main() {
	server := "127.0.0.1:8888"
	tcpAddr, err := net.ResolveTCPAddr("tcp4", server)
	conn, err := net.DialTCP("tcp", nil, tcpAddr)
	CheckError(err, "connected failed")
	go receiveInfo(conn)
	for {
		var content string
		fmt.Println("please input: ")
		fmt.Scanln(&content)
		send(conn, content)
	}
}
