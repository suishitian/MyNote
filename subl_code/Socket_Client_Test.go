package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
)

var (
	protocol  string
	operation string
	password  string
	username  string
	token     string
	content   string
)

type Info struct {
	Token     string
	Protocol  string
	Username  string
	Password  string
	Operation string
}

func CheckError(err error, str string) {
	if err != nil {
		fmt.Println(str)
	}
}

func receiveInfo(conn net.Conn) {
	buffer := make([]byte, 1024)
	conn.Read(buffer)
	index := bytes.IndexByte(buffer, 0)
	buffer1 := buffer[0:index]
	var rec Info
	err := json.Unmarshal(buffer1, &rec)
	if err == nil {
	}
	fmt.Print("received : ")
	fmt.Println(rec)
	if rec.Operation == "TokenSentBack" {
		token = rec.Token
		fmt.Println("recevied successfully " + rec.Token)
	} else {
		//fmt.Println("receive failed" + rec.Token)
	}
	//fmt.Println("rece over")
}

func send(conn net.Conn, buf []byte) {
	conn.Write(buf)
	fmt.Println("send successfully")
}
func main() {
	token = "nil"
	server := "127.0.0.1:8889"
	tcpAddr, err := net.ResolveTCPAddr("tcp4", server)
	conn, err := net.DialTCP("tcp", nil, tcpAddr)
	CheckError(err, "connected failed")

	for {
		//fmt.Println("loop begins")
		if token == "nil" {
			//fmt.Println("token==nil")
			fmt.Println("please input(have no token): ")
			fmt.Scanln(&protocol, &username, &password, &operation)
			m := Info{"nil", protocol, username, password, operation}
			packet, err := json.Marshal(m)
			if err == nil {
			}
			send(conn, packet)
		} else {
			//fmt.Println("token!=nil")
			fmt.Println("please input(had token): ")
			fmt.Scanln(&protocol, &username, &password, &operation)
			m := Info{token, protocol, username, password, operation}
			packet, err := json.Marshal(m)
			if err == nil {
			}
			send(conn, packet)
		}
		receiveInfo(conn)
	}
}
