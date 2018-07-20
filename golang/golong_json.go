package main

import (
	"encoding/json"
	"fmt"
)

type ABC struct {
	Name string
	Body string
	age  int64
}

func main() {

	m := ABC{"hehehe", "asdasd", 654654}
	var j []byte
	j, err := json.Marshal(m)
	if err != nil {
		print("ok")
	}
	fmt.Println(j)

	map1 := make(map[string]int)
	map1["haha"] = 20
	ss, ok := map1["haha1"]
	if !ok {
		fmt.Println("nono")
	} else {
		fmt.Println(ss)
	}

	var jj map[string]interface{}
	err = json.Unmarshal(j, &jj)
	aa, ok2 := jj["Name"]
	if ok2 {
		fmt.Println(aa)
	}
}
