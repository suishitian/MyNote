package main

import (
	"bytes"
	"fmt"
	"log"
	"net"
	"os"
)

func main() {

	//建立socket，监听端口
	netListen, err := net.Listen("tcp", "127.0.0.1:9998")
	CheckError(err)
	defer func(l net.Listener) {
		fmt.Println("关闭")
		l.Close()
	}(netListen)
	Log("Waiting for clients")
	for {

		conn, err := netListen.Accept()
		if err != nil {
			continue
		}
		Log(conn.RemoteAddr().String(), " 连接成功请求地址")
		go handleConnection(conn)
	}

}

//处理连接
func handleConnection(conn net.Conn) {
	buffer := make([]byte, 2048)
	Log("走了处理请求")
	for {
		Log("走的次数")
		n, err := conn.Read(buffer)
		if err != nil {
			Log(conn.RemoteAddr().String(), "连接错误的请求地址: ", err)
			return
		}
		Log(conn.RemoteAddr().String(), "这是啥数据:\n", string(buffer[:n]))
		if len(string(buffer[:n])) > 25 {
			sender(conn)
		}
	}
}
func sender(conn net.Conn) {
	Log("需要发送的xml")
	var buffer bytes.Buffer
	//var sl []string
	buffer.WriteString("<?xml version=\"1.0\" encoding=\"GBK\"?>")
	buffer.WriteString("<message>")
	buffer.WriteString("<head>")
	buffer.WriteString("<field name=\"ReceiveTime\">112823</field>")
	buffer.WriteString("<field name=\"ReceiveDate\">20151101</field>")
	buffer.WriteString("</head>")
	buffer.WriteString("<body>")
	buffer.WriteString("<field name=\"Host\">20151101</field>")
	buffer.WriteString("</body>")
	buffer.WriteString("</message>")
	Log(buffer.Bytes())
	Log("地址为===" + conn.RemoteAddr().String())
	//conn.Write([]byte(strings.Join(sl, "")))
	//-->使用数组的形式 得到byte也行 只不过看着没buffer这样的好
	// ar := []byte {1, 1,1, 1}
	//for i:= 0;i< len(buffer.Bytes()); i++ {
	//    ar = append(ar,buffer.Bytes()[i])
	//}
	//Log(ar)
	Log(buffer.String())
	conn.Write(buffer.Bytes())
	Log("send over")
}
func Log(v ...interface{}) {
	log.Println(v...)
}

func CheckError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
