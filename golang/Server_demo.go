package server

import (
	"fmt"
	"net"
	"reflect"
)

//======================================
func CheckError(err error, str string) {
	if err != nil {
		fmt.Println(str)
	}
}
func receive(conn net.Conn) []byte {
	buffer := make([]byte, 1024)
	conn.Read(buffer)
	fmt.Println(string(buffer))
	return buffer
}
func send(conn net.Conn, str string) {
	buffer := []byte(str)
	conn.Write(buffer)
	fmt.Println("send successfully")
}
func containsToken(obj interface{}, target interface{}) (bool, string) {
	targetValue := reflect.ValueOf(target)
	switch reflect.TypeOf(target).Kind() {
	case reflect.Slice, reflect.Array:
		for i := 0; i < targetValue.Len(); i++ {
			if targetValue.Index(i).Interface() == obj {
				return true, "have token"
			}
		}
	case reflect.Map:
		if targetValue.MapIndex(reflect.ValueOf(obj)).IsValid() {
			return true, "have token"
		}
	}

	return false, " there is no token ,need establish one"
}

//=====================================
type token string
type Server struct {
	IPaddr      string
	token_map   map[token]net.Conn
	chan_accept chan net.Conn
}

func getServer(addr string) *Server {
	cc := make(chan net.Conn)
	return &Server{addr, map[token]net.Conn{}, cc}
}

func (s *Server) handleClient() {
	for {
		conn := <-s.chan_accept
		go handleConn(conn)
	}
}
func (s *Server) handleConn(conn net.Conn) {
	bo, mess := containsToken(conn, s.token_map)
}

func (s *Server) handleAcceptedConn(conn net.Conn) {
	s.chan_accept <- conn
	//should return a message says cannot receive the data , chan is full.
}

func (s *Server) run() {
	fmt.Println("waiting for the client")
	netListen, err := net.Listen("tcp", s.IPaddr)
	defer func(listen net.Listener) {
		listen.Close()
		s.chan_accept.close()
	}(netListen)
	CheckError(err, "establish socket failed")
	go handleClient()
	for {
		conn, err := netListen.Accept()
		CheckError(err, "conn received failed")
		go handleAcceptedConn(conn)
	}
}
