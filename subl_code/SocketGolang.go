package main

import (
	"fmt"
	"net"
	"os"
	"time"
)

func main() {
	tcpAddr, err := net.ResolveTCPAddr("tcp4", ":8080")
	checkError(err)
	listener, err := net.ListenTCP("tcp", tcpAddr)
	checkError(err)
	for { //循环处理
		conn, err := listener.Accept()
		if err != nil {
			continue
		}
		go handleClient(conn) //创建一个goroutinue处理
	}
}

func handleClient(conn net.Conn) {
	defer conn.Close()
	daytime := time.Now().String()
	conn.Write([]byte(daytime)) // don't care about return value
	// we're finished with this client
}
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
