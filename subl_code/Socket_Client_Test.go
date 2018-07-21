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
	for {
		buffer := make([]byte, 1024)
		conn.Read(buffer)
		index := bytes.IndexByte(buffer, 0)
		buffer1 := buffer[0:index]
		var rec Info
		err := json.Unmarshal(buffer1, &rec)
		if err == nil {
		}
		if rec.Operation == "accept" {
			token = rec.Token
			fmt.Println(rec.Token)
		} else {
			fmt.Println(rec.Token)
		}
		//fmt.Println("sendBack: " + string(buffer))
	}
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
	//m := Info{"hehehe", 123}
	//var packet []byte
	//packet, err = json.Marshal(m)
	//if err == nil {
	//go receiveInfo(conn)
	for {
		if token == "nil" {
			fmt.Println("please input(have no token): ")
			fmt.Scanln(&protocol, &operation, &username, &password)
			m := Info{"nil", protocol, operation, username, password}
			packet, err := json.Marshal(m)
			if err == nil {
			}
			send(conn, packet)
		} else {
			fmt.Println("please input(had token): ")
			fmt.Scanln(&content)
			m := Info{token, protocol, content, username, password}
			packet, err := json.Marshal(m)
			if err == nil {
			}
			send(conn, packet)
		}
		receiveInfo(conn)
	}
}
