package main

import (
	"fmt"
)

var content string

func main() {
	for {
		fmt.Println("Please input something:")
		fmt.Scanln(&content)
		fmt.Println(content)
	}
}
